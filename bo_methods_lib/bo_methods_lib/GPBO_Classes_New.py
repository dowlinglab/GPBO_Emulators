import numpy as np
import random
from numpy.random import default_rng
import warnings
from datetime import datetime

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
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    WhiteKernel,
    ConstantKernel,
    DotProduct,
)
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

    Notes
    -------
    1 = A1 (Conventional GPBO, no obj scaling)
    2 = B1 (Conventional GPBO, ln obj scaling)
    3 = A2 (Emulator GPBO, independende approx. EI)
    4 = B2 (Emulator GPBO, log independence approx. EI)
    5 = C2 (Emulator GPBO, sparse grid integrated EI)
    6 = D2 (Emulator GPBO, monte carlo integrated EI)
    7 = A3 (Emulator GPBO, E[SSE] acquisition function)

    """

    # Ensure that only values 1 to 5 are chosen
    if Enum in range(1, 8) == False:
        raise ValueError("There are only seven options for Enum: 1 to 7")

    A1 = 1
    B1 = 2
    A2 = 3
    B2 = 4
    C2 = 5
    D2 = 6
    A3 = 7
    # Note use Method_name_enum.enum.name to call "A1"


class Kernel_enum(Enum):
    """
    Base class for kernel choices

    Notes
    -------
    1 = Matern 52
    2 = Matern 32
    3 = RBF
    """

    # Check that values are only 1 to 3
    if Enum in range(1, 4) == False:
        raise ValueError("There are only three options for Enum: 1 to 3")

    MAT_52 = 1
    MAT_32 = 2
    RBF = 3


class Gen_meth_enum(Enum):
    """
    The base class for any GPBO Method names

    Notes
    -------
    1 = LHS
    2 = Meshgrid
    """

    # Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 (LHS) to 2 (Meshgrid)")

    LHS = 1
    MESHGRID = 2


class Obj_enum(Enum):
    """
    The base class for any objective function

    Notes
    -------
    1 = SSE
    2 = ln(SSE)
    """

    # Check that values are only 1 to 2
    if Enum in range(1, 3) == False:
        raise ValueError("There are only two options for Enum: 1 (obj) to 2 (ln obj)")

    OBJ = 1
    LN_OBJ = 2


class Ep_enum(Enum):
    """
    The base class for any Method for calculating the decay of the exploration parameter

    Notes
    -------
    1 = Constant
    2 = Decay
    3 = Boyle
    4 = Jasrasaria
    """

    # Ensure that only values 1 to 5 are chosen
    if Enum in range(1, 4) == False:
        raise ValueError("There are only four options for Enum: 1 to 4")

    CONSTANT = 1
    DECAY = 2
    BOYLE = 3
    JASRASARIA = 4


class GPBO_Methods:
    """
    The base class for any GPBO Method

    Methods
    --------------
    __init__(*): Constructor method
    get_name_long(): Gets the shorthand name of the method that appears in the manuscript
    get_emulator(): Function to get emulator status based on method name
    get_obj(): Function to get objective function status based on method name
    get_sparse_mc(): Function to get sparse grid/Monte Carlo status based on method name
    """

    # Class variables and attributes

    def __init__(self, method_name):
        """
        Parameters
        ----------
        method_name: Method_name_enum Class instance, The name associated with the method being tested. Enum type
        """
        assert isinstance(
            method_name, Method_name_enum
        ), "method_name must be an instance of Method_name_enum"
        # Constructor method
        self.method_name = method_name
        self.emulator = self.get_emulator()
        self.obj = self.get_obj()
        self.report_name = self.get_name_long()
        self.sparse_grid, self.mc = self.get_sparse_mc()

    def get_name_long(self):
        """
        Gets the shorthand name of the method that appears in the manuscript

        Returns
        -------
        report_name: str, The shorthand name of the method that appears in the manuscript
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

        Returns
        --------
        emulator: bool, Status of whether the GP emulates the function directly
        """
        # Objective function uses emulator GP if class 2
        if not "1" in self.method_name.name:
            emulator = True
        else:
            emulator = False

        return emulator

    def get_obj(self):
        """
        Function to get objective function status based on method name

        Returns
        --------
        obj: class instance, Determines whether log scaling is used in the objective function
        """

        # Objective function is ln_obj if it includes the letter B
        if self.method_name.name == "B1":
            obj = Obj_enum(2)
        else:
            obj = Obj_enum(1)
        return obj

    def get_sparse_mc(self):
        """
        Function to get sparse grid and Monte Carlo status based on method name

        Returns
        --------
        sparse_grid: bool, Determines whether a sparse grid is used to evaluate the EI integral
        mc: bool, Determines whether an mc is used to evaluate the EI integral
        """
        # Sparse grid and MC default false
        sparse_grid = False
        mc = False

        # Check Emulator status
        if self.emulator == True:
            # Method 2C is Sparse Grid
            if "C" in self.method_name.name:
                sparse_grid = True
            # Method 2D is Monte Carlo
            elif "D" in self.method_name.name:
                mc = True

        return sparse_grid, mc


class CaseStudyParameters:
    """
    The base class for any GPBO Case Study and Evaluation

    Methods
    --------------
    __init__(*) : Constructor method

    """

    # Class variables and attributes

    def __init__(
        self,
        cs_name = "New_Case_Study", #Initialize Name
        ep0 = 1.0, 
        sep_fact = 1.0,
        normalize = True,
        kernel =  Kernel_enum(1),
        lenscl = None,
        outputscl = None,
        retrain_GP = 25,
        reoptimize_obj = 25,
        gen_heat_map_data = False,
        bo_iter_tot = 10,
        bo_run_tot = 1,
        save_data = False,
        DateTime = None,
        set_seed = 1,
        obj_tol = 1e-7,
        acq_tol = 1e-7,
        get_y_sse = False,
        w_noise = False,
    ):
        """
        Parameters
        ----------
        cs_name: string, default "New_Case_Study"
            The name associated with the case study being evaluated
        ep0: float or int, default 1.0
            The starting value for exploration bias parameter alpha
        sep_fact: float or int, default 1.0
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        normalize: bool, default True
            Determines whether feature data will be standardized (using sklearn RobustScaler)
        kernel: Kernel_enum, default Kernel_enum(3) (RBF Kernel)
            Determines which GP Kerenel to use
        lenscl: float, np.ndarray, or None, default None
            Value of the lengthscale hyperparameter - None if hyperparameters will be trained
        outputscl: float or None, default None
            Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_GP: int, default 25
            Number of times to (re)do GP training. Note, if zero, GP will not be trained at all and default/initial hyperparameters will be used
        reoptimize_obj: int, default 25
            Number of times to reoptimize ei/sse with different starting values. Note, 0 = 1 optimization
        gen_heat_map_data: bool, default False
            Determines whether validation data are generated to create heat maps
        bo_iter_tot: int, default 10
            Maximum number of BO iterations per restart
        bo_run_tot: int, default 1
            Total number of BO algorithm restarts
        save_data: bool, default False
            Determines whether ei data for argmax(ei) theta will be saved
        DateTime: str or None, default None (current date and time)
            Determines the date and time for the run
        set_seed: int or None, default 1
            Determines seed for randomizations. None if seed is random
        obj_tol: float, default 1e-7
            Objective difference at which to terminate algorithm (rho_1)
        acq_tol: float, default 1e-7
            Acquisition function value at which to terminate algorithm (rho_2)
        get_y_sse: bool, default False
            Determines whether to calculate the simulated y value when SSE is locally minimized
        w_noise: bool, default False
            Determines whether to include noise in the simulation data

        Raises
        ------
        AssertionError
            If any of the inputs (except cs_name) are not of the correct type
        Warning
            If cs_name is not a string
        """
        # Assert statements
        # Check for strings
        if not isinstance(cs_name, str) == True:
            warnings.warn(
                "cs_name will be converted to string if it is not an instance of CS_name_enum"
            )
        # Check for enum
        assert (
            isinstance(kernel, (Enum)) == True
        ), "kernel must be type Enum"  # Will figure this one out later
        # Check for float/int
        assert (
            all(isinstance(var, (float, int)) for var in [sep_fact, ep0]) == True
        ), "sep_fact and ep0 must be float or int"
        # Check for bool
        assert (
            all(
                isinstance(var, (bool))
                for var in [normalize, gen_heat_map_data, save_data]
            )
            == True
        ), "normalize, gen_heat_map_data, save_fig, and save_data must be bool"
        # Check for int
        assert (
            all(
                isinstance(var, (int))
                for var in [
                    bo_iter_tot,
                    bo_run_tot,
                    retrain_GP,
                    reoptimize_obj,
                ]
            )
            == True
        ), "bo_iter_tot, bo_run_tot, retrain_GP, and reoptimize_obj must be int"
        assert set_seed is None or (isinstance(set_seed, int) and set_seed >= 1), "set_seed must be int >= 1 or None"
        assert (
            isinstance(outputscl, (float, int)) or outputscl is None
        ), "outputscl must be float, int, or None"
        # Outputscl must be >0 if not None
        if outputscl is not None:
            assert outputscl > 0, "outputscl must be > 0 initially if it is not None"

        # Check lenscl, float, int, array, or None
        if isinstance(lenscl, list):
            lenscl = np.array(lenscl)

        assert isinstance(get_y_sse, bool), "get_y_sse must be bool"
        assert isinstance(w_noise, bool), "w_noise must be bool"

        assert (
            isinstance(lenscl, (float, int, np.ndarray)) or lenscl is None
        ), "lenscl must be float, int, np.ndarray, or None"
        if lenscl is not None:
            if isinstance(lenscl, (float, int)):
                assert lenscl > 0, "lenscl must be > 0 initially if lenscl is not None"
            else:
                assert all(
                    isinstance(var, (np.int64, np.float64, float, int))
                    for var in lenscl
                ), "All lenscl elements must float or int"
                assert all(
                    item > 0 for item in lenscl
                ), "lenscl elements must be > 0 initially if lenscl is not None"
        # Check for sep fact number between 0 and 1
        assert (
            0 < sep_fact <= 1
        ), "Separation factor must be between 0 and 1. Not including zero"
        # Check for > 0
        assert (
            all(var > 0 for var in [bo_iter_tot, bo_run_tot]) == True
        ), "bo_iter_tot and bo_run_tot must be > 0"
        # Check for >=0
        assert (
            all(var >= 0 for var in [retrain_GP, reoptimize_obj]) == True
        ), "retrain_GP and reoptimize_obj must be >= 0"
        # Check for str or None
        assert (
            isinstance(DateTime, (str)) == True or DateTime == None
        ), "DateTime must be str or None"
        assert (
            isinstance(acq_tol, (float, int)) and acq_tol >= 0
        ), "acq_tol must be a positive float or integer"
        assert (
            isinstance(obj_tol, (float, int)) and obj_tol >= 0
        ), "obj_tol must be a positive float or integer"

        if DateTime is None:
            DateTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Constructor method
        # Ensure name is a string
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
        self.acq_tol = acq_tol
        self.obj_tol = obj_tol
        self.get_y_sse = get_y_sse
        self.w_noise = w_noise


class Simulator:
    """
    The base class for differet simulators. Defines a simulation

    Methods
    --------------
    __init__(*): Constructor method
    __set_true_params(): Sets true parameter value array and the corresponding names based on parameter dictionary and indices to consider
    __grid_sampling(num_points, bounds): Generates Grid sampled data
    __lhs_sampling(num_points, bounds, seed): Design LHS Samples
    __create_param_data(num_points, bounds, gen_meth, seed): Generates data based off of bounds, and sampling scheme
    __vector_to_1D_array(array): Converts a vector to a 1D array
    gen_y_data(data, noise_mean, noise_std): Generates y data with noise
    gen_exp_data(num_x_data, gen_meth_x, set_seed=None, x_vals = None): Generates experimental data
    gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, gen_val_data): Generates simulation data
    gen_theta_vals(num_theta_data, rng): Generates parameter sets
    sim_data_to_sse_sim_data(method, sim_data, exp_data, sep_fact, gen_val_data)
    """

    def __init__(
        self,
        indices_to_consider,
        theta_ref,
        theta_names,
        bounds_theta_l,
        bounds_x_l,
        bounds_theta_u,
        bounds_x_u,
        noise_mean,
        noise_std,
        set_seed,
        calc_y_fxn,
        calc_y_fxn_args,
    ):
        """
        Parameters
        ----------
        indices_to_consider: list(int)
            The indices corresponding to which parameters are being guessed
        theta_ref: ndarray
            The array containing the true values of problem constants
        theta_names: list
            List of names of each parameter that will be plotted named by index w.r.t Theta_True
        bounds_theta_l: list
            Lower bounds of theta
        bounds_x_l: list
            Lower bounds of x
        bounds_theta_u: list
            Upper bounds of theta
        bounds_x_u: list
            Upper bounds of x
        noise_mean: float, int
            The mean of the noise
        noise_std: float, int, or None
            The standard deviation of the noise. If None, 1% of median of Y-exp will be used
        set_seed: int or None
            Determines seed for randomizations. None if seed is random
        calc_y_fxn: function
            The function to calculate ysim data with
        calc_y_fxn_args: dict
            Dictionary of arguments other than parameters and x to pass to calc_y_fxn

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        # Check for float/int
        assert isinstance(noise_mean, (float, int)), "noise_mean must be int or float"
        assert (
            isinstance(noise_std, (float, int)) or noise_std is None
        ), "noise_std must be int, float, or None"
        assert isinstance(set_seed, int) or set_seed is None, "Seed must be int or None"
        # Check for list or ndarray
        list_vars = [
            indices_to_consider,
            theta_ref,
            theta_names,
            bounds_theta_l,
            bounds_x_l,
            bounds_theta_u,
            bounds_x_u,
        ]
        assert (
            all(isinstance(var, (list, np.ndarray)) for var in list_vars) == True
        ), "indices_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must be list or np.ndarray"
        # Check for list lengths > 0
        assert (
            all(len(var) > 0 for var in list_vars) == True
        ), "indices_to_consider, theta_ref, theta_names, bounds_theta_l, bounds_x_l, bounds_theta_u, and bounds_x_u must have length > 0"
        # Check that bound_x and bounds_theta have same lengths
        assert len(bounds_theta_l) == len(bounds_theta_u) and len(bounds_x_l) == len(
            bounds_x_u
        ), "bounds lists for x and theta must be same length"
        # Check indeces to consider in theta_ref
        assert (
            all(0 <= idx <= len(theta_ref) - 1 for idx in indices_to_consider) == True
        ), "indeces to consider must be in range of theta_ref"
        assert (
            isinstance(calc_y_fxn_args, dict) or calc_y_fxn_args is None
        ), "calc_y_fxn_args must be dict or None"
        assert callable(
            calc_y_fxn
        ), "The argument 'calc_y_fxn' must be a callable (function) with 3 arguments."

        # Constructor method
        self.dim_x = len(bounds_x_l)
        self.dim_theta = len(
            indices_to_consider
        )  # Length of theta is equivalent to the number of indeces to consider
        self.indices_to_consider = indices_to_consider
        self.theta_ref = theta_ref
        self.theta_names = theta_names
        self.theta_true, self.theta_true_names = (
            self.__set_true_params()
        )  # Would this be better as a dictionary?
        self.bounds_theta = np.array([bounds_theta_l, bounds_theta_u])
        self.bounds_theta_reg = self.bounds_theta[
            :, self.indices_to_consider
        ]  # This is the theta_bounds for parameters we will regress
        self.bounds_x = np.array([bounds_x_l, bounds_x_u])
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.calc_y_fxn = calc_y_fxn
        self.calc_y_fxn_args = calc_y_fxn_args

        random_set_seed = random.randint(1, 1e8)

        self.rng_rand = np.random.default_rng()
        
        if set_seed is not None:
            self.rng_set = np.random.default_rng(set_seed)
            self.rng_exp = self.rng_set
        else:
            self.rng_set = self.rng_rand #np.random.default_rng(random_set_seed)
            self.rng_exp = np.random.default_rng(random_set_seed)
            
        #Ensure LHS for sim, val, and starting pts for EI will all be different
        if set_seed is not None:
            self.sim_seed = set_seed
            self.sim_x_seed = set_seed
            self.val_seed = set_seed + 1
            self.start_seed = set_seed + 2
        else:
            self.sim_seed = self.val_seed = self.start_seed = None
            self.sim_x_seed = random_set_seed

    def __set_true_params(self):
        """
        Sets true parameter value array and the corresponding names based on parameter dictionary and indices to consider

        Returns
        -------
        true_params: ndarray
            The true parameter of the model
        true_param_names: list(str)
            The names of the true parameter of the model
        """
        # Define theta_true and theta_true_names from theta_ref, theta_names, and indeces to consider
        true_params = self.theta_ref[self.indices_to_consider]
        true_param_names = [self.theta_names[idx] for idx in self.indices_to_consider]

        return true_params, true_param_names

    def __grid_sampling(self, num_points, bounds):
        """
        Generates grid sampled data

        Parameters
        ----------
        num_points: int
            Number of points to generate in each dimension, should be greater than # of dimensions
        bounds: ndarray
            Array containing upper and lower bounds of elements in each dimension.

        Returns
        ----------
        grid_data: np.ndarray
            (num_points)**bounds.shape[1] grid sample of data

        """
        # Generate mesh_grid data for theta_set in 2D
        # Define linspace for theta
        params = np.linspace(0, 1, num_points)
        # Define dimensions of parameter
        dimensions = bounds.shape[1]
        # Generate the equivalent of all meshgrid points
        df = pd.DataFrame(list(itertools.product(params, repeat=dimensions)))
        df2 = df.drop_duplicates()
        scaled_data = df2.to_numpy()
        # Normalize to bounds
        lower_bound = bounds[0]
        upper_bound = bounds[1]
        grid_data = scaled_data * (upper_bound - lower_bound) + lower_bound
        return grid_data

    def __lhs_sampling(self, num_points, bounds, rng):
        """
        Design LHS Samples

        Parameters
        ----------
        num_points: int
            Number of points in LHS, should be greater than # of dimensions
        bounds: np.ndarray
            Array containing upper and lower bounds of elements in LHS sample
        set_seed: int
            Seed of random generation

        Returns
        -------
        lhs_data: np.ndarray
            Array of LHS sampling points with length (num_points)
        """
        # Define number of dimensions
        dimensions = bounds.shape[1]
        # Define sampler
        #Note: "seed" in qmc.LatinHypercube will be deprecated after version 1.15.2, use rng if this becomes an issue
        sampler = qmc.LatinHypercube(d=dimensions, seed=rng)
        lhs_data = sampler.random(n=num_points)

        # Generate LHS data given bounds
        lhs_data = qmc.scale(
            lhs_data, bounds[0], bounds[1]
        )  # Using this because I like that bounds can be different shapes

        return lhs_data

    def __create_param_data(self, num_points, bounds, gen_meth, rng):
        """
        Generates data based off of bounds, and and generation scheme

        Parameters
        ----------
        num_points: int
            Number of data to generate
        bounds: np.ndarray
            Array of parameter bounds
        gen_meth: Gen_meth_enum
            ("LHS", "Meshgrid"). Determines whether data will be generated with an LHS or meshgrid
        set_seed: int
            Seed of random generation

        Returns
        --------
        data: np.ndarray
            An array of data

        Raises
        ------
        ValueError
            If gen_meth.value is not 1 or 2

        Notes
        ------
        Meshgrid generated data will output num_points in each dimension, LHS generates num_points of data
        """

        # Set dimensions
        dimensions = bounds.shape[
            1
        ]  # Want to do it this way to make it general for either x or theta parameters

        # Decide on a method to use based on gen_meth_value. LHS or Grid
        if gen_meth.value == 2:
            data = self.__grid_sampling(num_points, bounds)

        elif gen_meth.value == 1:
            # Generate LHS sample
            data = self.__lhs_sampling(num_points, bounds, rng)

        else:
            raise ValueError("gen_meth.value must be 1 or 2!")

        return data

    def __vector_to_1D_array(self, array):
        """
        Turns arrays that are shape (n,) into (n, 1) arrays

        Parameters
        ----------
        array: np.ndarray
            Array of n dimensions

        Returns
        -------
        array: np.ndarray
            If n > 1, return original array. Otherwise, return 2D array with shape (-1,n)
        """
        # If array is not 2D, give it shape (len(array), 1)
        if not len(array.shape) > 1:
            array = array.reshape(-1, 1)
        return array

    def gen_y_data(self, data, noise_mean, noise_std, rng, noise_std_pct = 0.01):
        """
        Creates simulated data based on the function self.calc_y_fxn

        Parameters
        ----------
        data: Data
            Parameter sets to generate y data for
        noise_mean: float, int
            The mean of the noise
        noise_std: float, int, None
            The standard deviation of the noise

        Returns
        -------
        y_data: np.ndarray The simulated y training data
        """
        if noise_std is None:
            assert isinstance(noise_std_pct, (float, int)) and noise_std_pct >= 0, "noise_std_pct must be positive float or int"

        # Define an array to store y values in
        y_data = []
        # Get number of points
        len_points = data.get_num_theta()
        # Loop over all theta values
        for i in range(len_points):
            # Create model coefficient from true space substituting in the values of param_space at the correct indeces
            model_coefficients = self.theta_ref.copy()
            # Replace coefficients a specified indeces with their theta_val counterparts
            model_coefficients[self.indices_to_consider] = data.theta_vals[i]
            # Create y data coefficients
            y_data.append(
                self.calc_y_fxn(
                    model_coefficients, data.x_vals[i], self.calc_y_fxn_args
                )
            )

        # Convert list to array and flatten array
        y_data = np.array(y_data).flatten()

        # Creates noise values with a certain stdev and mean from a normal distribution
        # If noise is none
        if noise_std is None:
            # Set the noise as 1% of the median as a default. 
            if not math.isclose(np.median(y_data),0):
                noise_std = np.abs(np.median(y_data)) * noise_std_pct
            #If the median value is 0, use 1% of the mean as the default.
            elif not math.isclose(np.mean(y_data),0):
                noise_std = np.abs(np.mean(y_data)) * noise_std_pct
            #If both values are zero, Use 1% of the abs max value
            else:
                noise_std = np.max(np.abs(y_data)) * noise_std_pct
            #Set temp noise to the noise value that was just generated. This value is only used if gen_exp_data is called
            self.temp_noise = noise_std
        else:
            noise_std = noise_std

        noise = rng.normal(size=len(y_data), loc=noise_mean, scale=noise_std)
        # print(noise.flatten())

        # Add noise to data
        y_data = y_data + noise

        return y_data

    def gen_exp_data(self, num_x_data, gen_meth_x, x_vals=None, noise_std_pct = 0.01):
        """
        Generates experimental data in an instance of the Data class

        Parameters
        ----------
        num_x_data: int
            Number of experiments
        gen_meth_x: bool
            Whether to generate X data with LHS or grid method
        set_seed: int or None, default None
            Seed with which t0 generate experimental data. None sets the seed to the class seed
        x_vals: np.ndarray or None, default None
            X values to use for experimental data. If None, x_vals will be generated based on bounds and num_x_data
        noise_std_pct: float or int, default 0.01
            Percentage of the mean of the y data to use as the standard deviation of the noise

        Returns
        --------
        exp_data: Data
            Experimental x and y data along with parameter bounds

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        ValueError
            If num_x_data is not a positive integer

        Notes:
        ------
        Warning: This function will not generate exactly the same values of y when repeatedly called, even with the same seed. 
        """
        assert x_vals is None or isinstance(x_vals, np.ndarray), "x_vals must be np.ndarray or None"

        assert isinstance(noise_std_pct, (int,float)) and noise_std_pct >= 0, "noise_std_pct must be a positive int/float"
        # check that num_data > 0
        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError("num_x_data must be a positive integer")

        # Create x vals based on bounds and num_x_data if x_vals are not specified
        #For data generation we always want instances of exp_data to be reproduceable, so sim_x_seed and rng_exp are used
        if x_vals is None:
            x_vals = self.__vector_to_1D_array(
                self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, self.sim_x_seed)
            )
        else:
            x_vals = self.__vector_to_1D_array(x_vals)
    
        # Reshape theta_true to correct dimensions and stack it once for each xexp value
        theta_true = self.theta_true.reshape(1, -1)
        theta_true_repeated = np.vstack([theta_true] * len(x_vals))
        # Create exp_data class and add values
        exp_data = Data(
            theta_true_repeated,
            x_vals,
            None,
            None,
            None,
            None,
            None,
            None,
            self.bounds_theta_reg,
            self.bounds_x,
            None,
        )
        # Generate y data for exp_data calss instance
        #We will always use the set_rng for data generation
        exp_data.y_vals = self.gen_y_data(exp_data, self.noise_mean, self.noise_std, self.rng_exp, noise_std_pct = noise_std_pct)

        #Set simulator noise after exp_data is generated if self.noise_std is None
        if self.noise_std == None:
            self.noise_std = self.temp_noise

        return exp_data

    def gen_sim_data(
        self,
        num_theta_data,
        num_x_data,
        gen_meth_theta,
        gen_meth_x,
        sep_fact,
        set_seed = None,
        gen_val_data=False,
        x_vals = None,
        w_noise = False,
    ):
        """
        Generates simulated data in an instance of the Data class

        Parameters
        ----------
        num_theta_data: int
            Number of parameter sets
        num_x_data: int
            Number of experiments
        gen_meth_theta: bool
            Whether to generate theta data with LHS or grid method
        gen_meth_x: bool
            Whether to generate X data with LHS or grid method
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None, default None
            Optional seed to generate initial LHS training data with. If None, seed will be the seed of the class
        gen_val_data: bool, default False
            Whether validation data (no y vals) or simulation data (has y vals) will be generated
        x_vals: np.ndarray or None, default None
            X values to use for simulation data. If None, x_vals will be generated based on bounds and num_x_data
        w_noise: bool, default False
            Whether to generate data with noise

        Returns
        --------
        sim_data: Data
            Simulated x and y data along with parameter bounds

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        ValueError
            If num_theta_data or num_x_data are not a positive integer or gen_val is not a boolean
        Warning
            If more than 5000 points are generated

        Notes:
        -------
        Warning: This function will not generate exactly the same values of y when repeatedly called, even with the same seed. 
        """
        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"

        #Random if rng is not set, otherwise set by seed of simulator
        rng = self.rng_set

        #For 
        if gen_val_data == False and self.sim_seed is not None:
            seed_theta = self.sim_seed
            seed_x = self.sim_seed
        elif gen_val_data == True and self.sim_seed is not None:
            seed_theta = self.val_seed
            seed_x = self.sim_seed
        else:
            seed_theta = None
            seed_x = self.sim_x_seed #For data generation we always want x to be the same
        
        #Set the theta seed to the given seed if one is provided
        if set_seed is not None:
            seed_theta = set_seed

        if isinstance(gen_val_data, bool) == False:
            raise ValueError("gen_val_data must be bool")

        # Chck that num_data > 0
        if num_theta_data <= 0 or isinstance(num_theta_data, int) == False:
            raise ValueError("num_theta_data must be a positive integer")

        if num_x_data <= 0 or isinstance(num_x_data, int) == False:
            raise ValueError("num_x_data must be a positive integer")

        # Set bounds on theta which we are regressing given bounds_theta and indeces to consider
        # X data we always want the same between simulation and validation data
        if x_vals is None:
            x_data = self.__vector_to_1D_array(
                self.__create_param_data(num_x_data, self.bounds_x, gen_meth_x, seed_x)
            )
        else:
            x_data = self.__vector_to_1D_array(x_vals)

        # Infer how many times to repeat theta and x values given whether they were generated by LHS or a meshgrid
        # X and theta repeated at least once per time the other is generated
        repeat_x = num_theta_data
        repeat_theta = len(x_data)

        # If using a meshgrid this number is exponentiated by the number of dimensions of itself
        if gen_meth_theta.value == 2:
            repeat_x = num_theta_data ** (self.dim_theta)
        if gen_meth_x.value == 2:
            repeat_theta = num_x_data ** (self.dim_x)

        # Warn user if >5000 pts generated
        if repeat_x * repeat_theta > 5000:
            warnings.warn("More than 5000 points will be generated!")

        # Generate all rows of simulation data
        sim_data = Data(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.bounds_theta_reg,
            self.bounds_x,
            sep_fact,
        )

        # Generate simulation data theta_vals and create instance of data class
        sim_theta_vals = self.__vector_to_1D_array(
            self.__create_param_data(
                num_theta_data, self.bounds_theta_reg, gen_meth_theta, seed_theta
            )
        )

        # Add repeated theta_vals and x_data to sim_data
        sim_data.theta_vals = np.repeat(sim_theta_vals, repeat_theta, axis=0)
        sim_data.x_vals = np.vstack([x_data] * repeat_x)

        # Add y_vals for sim_data
        if w_noise == False:
            #Default to noiseless training data
            sim_data.y_vals = self.gen_y_data(sim_data, self.noise_mean, 0, rng)
        else:
            # Generate train data with noise if some noise is specified
            sim_data.y_vals = self.gen_y_data(sim_data, self.noise_mean, self.noise_std, rng)

        return sim_data

    def gen_theta_vals(self, num_theta_data, rng_seed = None):
        """
        Generates parameter sets for an instance of the Data class

        Parameters
        ----------
        num_theta_data: int
            Number of parameter sets

        Returns
        --------
        theta_vals: np.ndarray
            Generated parameter sets

        Raises
        ------
        AssertionError
            If num_theta_data is not a positive integer
        Warning
            If more than 5000 points are generated
        """
        #Ensures seed will never be the same as the data generation seeds (always higher)
        if rng_seed == None:
            rng_seed = self.start_seed
        else:
            rng_seed += self.start_seed

        assert (
            isinstance(num_theta_data, int) and num_theta_data > 0
        ), "num_theta_data must be int > 0"
        gen_meth_theta = Gen_meth_enum(1)

        # Warn user if >5000 pts generated
        if num_theta_data > 5000:
            warnings.warn("More than 5000 points will be generated!")

        # Generate simulation data theta_vals and create instance of data class
        theta_vals = self.__vector_to_1D_array(
            self.__create_param_data(
                num_theta_data, self.bounds_theta_reg, gen_meth_theta, rng_seed
            )
        )

        return theta_vals

    def sim_data_to_sse_sim_data(
        self, method, sim_data, exp_data, sep_fact, y_to_sse=False
    ):
        """
        Creates objective function simulation data based on state points, parameter sets, the GPBO method, and self.calc_y_fxn

        Parameters
        ----------
        method: GPBO_Methods
            Fully defined methods class which determines which method will be used
        sim_data: Data
            Class containing at least the theta_vals, x_vals, and y_vals for simulation
        exp_data: Data
            Class containing at least the x_data and y_data for the experimental data
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        y_to_sse: bool, default False
            Whether sim_data.y_vals will be set as y (True) or sse(y) (False)

        Returns
        --------
        sim_sse_data: np.ndarray
            Objective function data generated from y_vals

        Raises
        ------
        AssertionError
            If sep_fact is not between 0 and 1
        ValueError
            If y_to_sse is not a boolean
        """

        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"

        if isinstance(y_to_sse, bool) == False:
            raise ValueError("y_to_sse must be bool")

        # Find length of theta and x in data arrays
        len_theta = sim_data.get_num_theta()
        len_x = exp_data.get_num_x_vals()

        # Q: For this dataset does it make more sense to have all theta and x values or just the unique thetas and x values?
        # A: Just the unique ones. No need to store extra data if we won't use it and it will be saved somewhere else regardless
        # Assign unique theta indeces and create an array of them
        unique_indexes = np.unique(sim_data.theta_vals, axis=0, return_index=True)[1]
        unique_theta_vals = np.array(
            [sim_data.theta_vals[index] for index in sorted(unique_indexes)]
        )
        # Add the unique theta_vals and exp_data x values to the new data class instance
        sim_sse_data = Data(
            unique_theta_vals,
            exp_data.x_vals,
            None,
            None,
            None,
            sim_data.sse,
            sim_data.sse_var,
            sim_data.acq,
            self.bounds_theta,
            self.bounds_x,
            sep_fact,
        )

        if y_to_sse == False and sim_data.y_vals is not None:
            # Define all y_sims
            y_sim = sim_data.y_vals

            # Reshape y_sim into n_theta rows x n_x columns
            indices = np.arange(0, len_theta, len_x)
            n_blocks = len(indices)
            # Slice y_sim into blocks of size len_x and calculate squared errors for each block
            y_sim_resh = y_sim.reshape(n_blocks, len_x)
            block_errors = (y_sim_resh - exp_data.y_vals[np.newaxis, :]) ** 2
            # Sum squared errors for each block
            sum_error_sq = np.sum(block_errors, axis=1)
            # objective function only log if using 1B
            if method.obj.value == 2:
                #Set a minimum error to avoid log(0)
                sum_error_sq[sum_error_sq < 1e-16] = 1e-16
                sum_error_sq = np.log(sum_error_sq)  # Scaler

            # Add y_values to data class instance
            sim_sse_data.y_vals = sum_error_sq

        return sim_sse_data


class Data:
    """
    The base class for any Data used in this workflow
    Parameters

    Methods
    --------------
    __init__(*): Constructor method
    __get_unique(all_vals): Gets unique instances of a certain type of data
    __vector_to_1D_array(array): Turns arrays that are shape (n,) into (n, 1) arrays
    get_unique_theta(): Defines the unique parameter sets from self.theta_vals
    get_unique_x(): Defines the unique state point data from self.x_vals
    get_num_theta(): Defines the total number of parameter sets (self.theta_vals)
    get_dim_theta(): Defines the total dimensions of the parameter sets (self.theta_vals)
    get_num_x_vals(): Defines the total number of state points (self.x_vals)
    get_dim_x_vals(): Defines the total dimensions of the state points (self.x_vals)
    train_test_idx_split(): Splits data into training and testing data
    """

    # Class variables and attributes

    def __init__(
        self,
        theta_vals,
        x_vals,
        y_vals,
        gp_mean,
        gp_var,
        sse,
        sse_var,
        acq,
        bounds_theta,
        bounds_x,
        sep_fact
    ):
        """
        Parameters
        ----------
        theta_vals: np.ndarray
            The array of parameter sets
        x_vals: np.ndarray
            Experimental state points (x data)
        y_vals: np.ndarray
            Experimental y data
        gp_mean: np.ndarray
            GP mean prediction values associated with theta_vals and x_vals
        gp_var: np.ndarray
            GP variance prediction values associated with theta_vals and x_vals
        sse: np.ndarray
            GP based sum of squared error values associated with theta_vals and x_vals
        sse_var: np.ndarray
            GP based variance of sum of squared error values associated with theta_vals and x_vals
        acq: np.ndarray
            Acquisition function values associated with theta_vals and x_vals
        bounds_theta: np.ndarray
            Bounds of theta
        bounds_x: np.ndarray
            Bounds of x
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        list_vars = [theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, acq]
        assert all(
            isinstance(var, np.ndarray) or var is None for var in list_vars
        ), "theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, and ei must be np.ndarray, or None"
        assert (
            isinstance(sep_fact, (float, int)) or sep_fact is None
        ), "Separation factor must be float or int > 0 or None (exp_data)"
        if sep_fact is not None:
            assert (
                0 < sep_fact <= 1
            ), "sep_fact must be > 0 and less than or equal to 1!"
        # Constructor method
        self.theta_vals = theta_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.gp_covar = None  # This is calculated later
        self.sse = sse
        self.sse_var = sse_var
        self.sse_covar = None  # This is calculated later
        self.acq = acq
        self.bounds_theta = bounds_theta
        self.bounds_x = bounds_x
        self.sep_fact = sep_fact

    def __get_unique(self, all_vals):
        """
        Gets unique instances of a certain type of data

        Parameters
        -----------
        all_vals: np.ndarray
            Array of parameters with duplicates

        Returns
        --------
        unique_vals: np.ndarray
            Array of parameters without duplicates
        """
        # Get unique indecies and use them to get the values
        unique_indexes = np.unique(all_vals, axis=0, return_index=True)[1]
        unique_vals = np.array([all_vals[index] for index in sorted(unique_indexes)])

        return unique_vals

    def __vector_to_1D_array(self, array):
        """
        Turns arrays that are shape (n,) into (n, 1) arrays

        Parameters
        ----------
        array: np.ndarray
            Array of n dimensions

        Returns
        -------
        array: np.ndarray
            If n > 1, return original array. Otherwise, return 2D array with shape (-1,n)
        """
        # If array is not 2D, give it shape (len(array), 1)
        if not len(array.shape) > 1:
            array = array.reshape(-1, 1)
        return array

    def get_unique_theta(self):
        """
        Defines the unique parameter sets from self.theta_vals

        Returns
        --------
        unique_theta_vals: np.ndarray
            Array of unique parameter sets

        Raises
        ------
        AssertionError
            If any self.theta_vals is not defined
        """
        assert self.theta_vals is not None, "self.theta_vals must be defined"
        # Get unique indecies and use them to get the values
        unique_theta_vals = self.__get_unique(self.theta_vals)
        return unique_theta_vals

    def get_unique_x(self):
        """
        Defines the unique state point data from self.x_vals

        Returns
        --------
        unique_x_vals: np.ndarray
            Array of unique state points

        Raises
        ------
        AssertionError
            If self.x_vals is not defined
        """
        assert self.x_vals is not None, "self.x_vals must be defined"
        # Get unique indecies and use them to get the values
        unique_x_vals = self.__get_unique(self.x_vals)
        return unique_x_vals

    def get_num_theta(self):
        """
        Defines the total number of parameter sets (self.theta_vals)

        Returns
        -------
        num_theta_data: int
            The number of parameter sets (self.theta_vals)

        Raises
        ------
        AssertionError
            If self.theta_vals is not defined
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        num_theta_data = len(self.theta_vals)

        return num_theta_data

    def get_dim_theta(self):
        """
        Defines the total dimensions of the parameter sets (self.theta_vals)

        Returns
        -------
        dim_theta_data: int
            The cardinality of the parameter sets (self.theta_vals)

        Raises
        ------
        AssertionError
            If self.theta_vals is not defined
        """
        assert self.theta_vals is not None, "self.theta_vals must be defined"
        if len(self.theta_vals) == 1:
            theta_vals = self.theta_vals.reshape(1, -1)
        else:
            theta_vals = self.theta_vals

        dim_theta_data = theta_vals.shape[1]

        return dim_theta_data

    def get_num_x_vals(self):
        """
        Defines the total number of state point data (self.x_vals)

        Returns
        -------
        num_x_data: int
            The number of state points (self.x_vals)

        Raises
        ------
        AssertionError
            If self.x_vals is not defined
        """
        assert self.x_vals is not None, "self.x_vals must be defined"
        # Length is the number of data
        num_x_data = len(self.x_vals)

        return num_x_data

    def get_dim_x_vals(self):
        """
        Defines the total dimensions of state point data (self.x_vals)

        Returns
        -------
        dim_x_data: int
            The cardinality of state point data (self.x_vals)

        Raises
        ------
        AssertionError
            If self.x_vals is not defined
        """
        assert self.x_vals is not None, "x_vals must be defined"
        # Get dim of x data
        dim_x_data = self.__vector_to_1D_array(self.x_vals).shape[1]

        return dim_x_data

    def train_test_idx_split(self, rng_seed = None):
        """
        Splits data indices into training and testing indices

        Returns
        --------
        train_idx: np.ndarray
            The training theta data identifiers
        test_idx: np.ndarray
            The testing theta data identifiers

        Raises
        ------
        AssertionError
            If self.sep_fact or self.theta_vals are not defined

        Notes
        -----
        The training and testing data is split such that the number train_data is always rounded up. Ensures there is always training data

        """
        assert (
            self.sep_fact is not None
        ), "Data must have a separation factor that is not None!"
        assert self.theta_vals is not None, "data must have theta_vals"
        assert isinstance(rng_seed, int) or rng_seed is None, "rng_seed must be int or None"

        #Save the seed used to shuffle and split the data + create rng w/ this seed. Default to no seed
        self.seed = rng_seed
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
        else:
            rng = np.random.default_rng()

        # Find number of unique thetas and calculate length of training data
        len_theta = len(self.get_unique_theta())
        len_train_idc = int(
            np.ceil(len_theta * self.sep_fact)
        )  # Ensure there will always be at least one training point by using np.ceil

        # Create an index for each theta
        all_idx = np.arange(0, len_theta)
        
        # Shuffle all_idx data in such a way that theta values will be randomized
        rng.shuffle(all_idx)
        
        # Set train test indeces
        train_idx = all_idx[:len_train_idc]
        test_idx = all_idx[len_train_idc:]

        return train_idx, test_idx
class GP_Emulator:
    """
    The base class for Gaussian Processes used in this workflow

    Methods
    --------------
    __init__(*) : Constructor method
    get_num_gp_data(): Defines the total number of all simulated accessible to the GP
    __set_lenscl_guess(lb, ub): Sets the lengthscale guess
    __set_white_kern(lb, ub): Sets the white kernel guess
    __set_outputscl(lb, ub): Sets the outputscale (tau) guess
    set_gp_model_data(): Sets training data for the GP model data
    __init_hyper_parameters(retrain_count): Initializes hyperparameters for the GP model
    set_gp_model(retrain_count): Builds the GP model
    train_gp(): Trains the GP model
    __eval_gp_mean_var(data): Evaluates the mean and variance of the GP model
    eval_gp_mean_var_misc(misc_data, featurized_misc_data, covar=False): Evaluates the mean and variance of the GP model for miscellaneaous data
    eval_gp_mean_var_test(covar=False): Evaluates the mean and variance of the GP model for testing data
    eval_gp_mean_var_val(covar=False): Evaluates the mean and variance of the GP model for validation data
    eval_gp_mean_var_cand(covar=False): Evaluates the mean and variance of the GP model for candidate data
    """

    # Class variables and attributes

    def __init__(
        self,
        gp_sim_data,
        gp_val_data,
        cand_data,
        kernel,
        lenscl,
        noise_std,
        outputscl,
        retrain_GP,
        set_seed,
        normalize,
        __feature_train_data,
        __feature_test_data,
        __feature_val_data,
        __feature_cand_data,
    ):
        """
        Parameters
        ----------
        gp_sim_data: Data
            All simulation data for the GP
        gp_val_data: Data
            The validation data for the GP. None if not saving validation data
        cand_data: Data
            Candidate theta value for evaluation with GPBO_Driver.opt_with_scipy()
        kernel: Kernel_enum
            Determines which GP Kerenel to use
        lenscl: float or None
            Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int
            The standard deviation of the noise
        outputscl: float or None
            Determines value of outputscale
        retrain_GP: int
            Number of times to (re)do GP training. If 0, no training is done and default/initial values are used
        set_seed: int or None
            Random seed
        normalize: bool
            Determines whether data is standardized (using the sklearn RobustScaler)
        __feature_train_data: np.ndarray
            The feature data for the training data in ndarray form
        __feature_test_data: np.ndarray
            The feature data for the testing data in ndarray form
        __feature_val_data: np.ndarray
            The feature data for the validation data in ndarray form
        __feature_cand_data: np.ndarray
            The feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        # Assert statements
        # Check for int/float
        assert (
            isinstance(outputscl, (float, int)) or outputscl is None
        ), "outputscl must be float, int, or None"
        # Check that set values for outputscl are in range
        if outputscl is not None:
            assert (
                100 > outputscl > 1e-5
            ), "outputscl must be in range [1e-5,1e3] if it is not None"

        # Check lenscl, float, int, array, or None
        if isinstance(lenscl, list):
            lenscl = np.array(lenscl)
        assert (
            isinstance(lenscl, (float, int, np.ndarray)) or lenscl is None
        ), "lenscl must be float, int, np.ndarray, or None"
        # Check that set values for lenscl are in range
        if lenscl is not None:
            if isinstance(lenscl, (float, int)):
                assert (
                    1000 > lenscl > 1e-5
                ), "lenscl must be in range [1e-5,1e3] if lenscl is not None"
            else:
                assert all(
                    isinstance(var, (np.int64, np.float64, float, int))
                    for var in lenscl
                ), "All lenscl elements must float or int"
                assert all(
                    1000 > item > 1e-5 for item in lenscl
                ), "lenscl elements must be in range [1e-5,1e3] if lenscl is not None"
                lenscl = lenscl.astype(np.float64)  # Convert all guesses to float64

        assert isinstance(normalize, bool), "normalize must be bool"
        assert (
            isinstance(retrain_GP, int) == True and retrain_GP >= 0
        ), "retrain_GP must be int greater than or equal to 0"
        # Check for Enum
        assert isinstance(kernel, Enum) == True, "kernel must be type Enum"
        # Check for instance of Data class or None
        assert (
            isinstance(gp_sim_data, (Data)) == True or gp_sim_data == None
        ), "gp_sim_data must be an instance of the Data class or None"
        assert (
            isinstance(gp_val_data, (Data)) == True or gp_val_data == None
        ), "gp_sim_data must be an instance of the Data class or None"

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
        # If normalize, create the scalers
        if normalize == True:
            self.scalerX = RobustScaler(unit_variance=True)
            self.scalerY = RobustScaler(unit_variance=True)
        self.__feature_train_data = None  # Added using child class
        self.__feature_test_data = None  # Added using child class
        self.__feature_val_data = None  # Added using child class
        self.__feature_cand_data = None  # Added using child class

        self.rng_rand = np.random.default_rng()
        if self.seed != None:
            self.rng_set = np.random.default_rng(self.seed)
        else:
            self.rng_set = self.rng_rand

    def get_num_gp_data(self):
        """
        Defines the total number of data the GP will have access to

        Returns
        -------
        num_data: int
            The number of data the GP will have access to

        Raises
        ------
        AssertionError
            If self.gp_sim_data is not an instance of the Data class
        """
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data class"
        # Number of available gp data determined by number of sim data
        num_gp_data = int(self.gp_sim_data.get_num_theta())

        return num_gp_data

    def bounded_parameter(self, low, high, initial_value):
        """
        Creates a bounded parameter for the GP model

        Parameters
        ----------
        low: float, int
            Lower bound of the parameter
        high: float, int
            Upper bound of the parameter
        initial_value: float, int, np.ndarray
            Initial value of the parameter

        Returns
        -------
        parameter: gpflow.Parameter
            The bounded parameter

        Raises
        ------
        AssertionError
            If the lower bound is higher than the upper bound
        """
        assert isinstance(low, (float, int)), "low must be float or int"
        assert isinstance(high, (float, int)), "low must be float or int"
        assert isinstance(
            initial_value, (float, int, np.ndarray)
        ), "initial_value must be float, int, or np.ndarray of shape (n,)"
        if isinstance(initial_value, np.ndarray):
            assert len(initial_value.shape) <= 1, "initial_value must be a scalar or 1D array"
        assert low < high, "low must be less than high"
        sigmoid = tfb.Sigmoid(
            low=tf.cast(low, dtype=tf.float64), high=tf.cast(high, dtype=tf.float64)
        )
        return gpflow.Parameter(initial_value, transform=sigmoid, dtype=tf.float64)

    def __set_lenscl_guess(self, lb, ub):
        """
        Sets the lengthscale guess for the GP model

        Parameters
        ----------
        lb: float, int
            Lower bound of the lengthscale
        ub: float, int
            Upper bound of the lengthscale

        Returns
        --------
        lenscl_guess: np.ndarray
            The intial lengthscale of the GP model

        Raises
        ------
        AssertionError
            If self.train_data_init is not an array
        """
        rng = self.rng_set
        # Set lenscl bounds using the original training data to ensure distance
        # Between min and max lengthscales does not collapse as iterations progress
        assert isinstance(
            self.train_data_init, np.ndarray
        ), "self.train_data_init must be an array"
        if self.normalize:
            org_scalerX = RobustScaler(unit_variance=True)
            points = org_scalerX.fit_transform(self.train_data_init)
        else:
            points = self.train_data_init

        # Compute pairwise differences for each column
        pairwise_diffs = np.abs(
            points[:, :, None] - points[:, :, None].transpose(0, 2, 1)
        )
        # Compute Euclidean distances
        euclidean_distances = np.sqrt(np.sum(pairwise_diffs**2, axis=1))
        # Set diagonal elements (distance between the same point) to infinity
        np.fill_diagonal(euclidean_distances, np.inf)
        euclidean_distances = np.ma.masked_invalid(euclidean_distances)
        # Find the smallest/largest distance for each column and ensure it is within the bounds
        min_distance = np.min(euclidean_distances, axis=0)
        max_distance = np.max(euclidean_distances, axis=0)

        lb_array = np.ones(len(min_distance)) * lb
        ub_array = np.ones(len(max_distance)) * ub
        lower = np.maximum(min_distance, lb_array)
        upper = np.minimum(max_distance, ub_array)

        lenscl_guess = rng.uniform(lower, upper, size=len(max_distance))
        return lenscl_guess

    def __set_white_kern(self, lb, ub):
        """
        Sets the white kernel value guess for the GP model

        Parameters
        ----------
        lb: float, int
            Lower bound of the white kernel
        ub: float, int
            Upper bound of the white kernel

        Returns
        --------
        noise_guess: float
            The initial white noise variance for the GP model
        """
        # Set the noise guess or allow gp to tune the noise parameter
        if self.normalize:
            self.scalerY.fit(self.train_data.y_vals.reshape(-1, 1))
            sclr = np.float64(self.scalerY.scale_)
        else:
            sclr = 1.0

        if self.noise_std is not None:
            # If we know the noise, use it
            noise_guess = float((self.noise_std / sclr) ** 2)

        else:
            # Otherwise, set the guess as 1% the taining data median
            if not math.isclose(np.median(self.gp_sim_data.y_vals),0):
                data_mean = np.abs(np.median(self.gp_sim_data.y_vals))
            elif not math.isclose(np.mean(self.gp_sim_data.y_vals),0):
                data_mean = np.abs(np.mean(self.gp_sim_data.y_vals))
            else:
                data_mean = np.max(np.abs(self.gp_sim_data.y_vals))
            noise_guess = np.float64(data_mean * 0.01 / sclr) ** 2

        if not lb < noise_guess < ub:
            noise_guess = 1.0

        return noise_guess

    def __set_outputscl(self, lb, ub):
        """
        Set the initial output scale of the model

        Parameters
        ----------
        lb: float, int
            Lower bound of the output scale
        ub: float, int
            Upper bound of the output scale

        Returns
        -------
        tau: float
            Initial output scale guess for the GP model

        Notes
        ------
        Need to have training data before using this function
        """

        # Set outputscl kernel to be optimized based on guess if desired
        if self.outputscl == None:
            train_y = self.train_data.y_vals.reshape(-1, 1)
            if self.normalize:
                scl_y = self.scalerY.fit_transform(train_y)
            else:
                scl_y = train_y

            c_guess = sum(scl_y.flatten() ** 2) / len(scl_y)
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
        data: tuple(np.ndarrays, len=2)
            The feature and output training data for the GP model

        Raises
        ------
        AssertionError
            If self.feature_train_data or self.train_data.y_vals are not defined
        """
        assert (
            self.feature_train_data is not None
        ), "self.feature_train_data must be defined"
        assert (
            self.train_data.y_vals is not None
        ), "self.train_data.y_vals must be defined"
        # Set new model data
        # Preprocess Training data
        if self.normalize == True:
            # Update scaler to be the fitted scaler. This scaler will change as the training data is updated
            # Scale training data if necessary
            ft_td_scl = self.scalerX.fit_transform(self.feature_train_data)
            y_td_scl = self.scalerY.fit_transform(self.train_data.y_vals.reshape(-1, 1))
        else:
            ft_td_scl = self.feature_train_data
            y_td_scl = self.train_data.y_vals.reshape(-1, 1)
        data = (ft_td_scl, y_td_scl)
        return data

    def __init_hyper_parameters(self, retrain_count):
        """
        Initializes hyperparameters for the GP model

        Parameters
        ----------
        retrain_count: int
            The number of times the GP will be (re)trained

        Returns
        --------
        lenscls: np.ndarray
            The initial lengthscale of the GP model
        tau: float
            The initial output scale guess for the GP model
        white_var: float
            The initial white noise variance for the GP model
        """
        tf.compat.v1.get_default_graph()
        rng = self.rng_set
        gpflow.config.set_default_float(np.float64)

        # Set bounds for hyperparameters
        lenscl_bnds = [0.00001, 1000.0]
        var_bnds = [0.00001, 100.0]
        white_var_bnds = [0.00001, 10.0]

        # Get X and Y Data
        data = self.set_gp_model_data()
        x_train, y_train = data

        if isinstance(self.lenscl, np.ndarray):
            lenscls = self.bounded_parameter(
                lenscl_bnds[0], lenscl_bnds[1], self.lenscl
            )
        elif isinstance(self.lenscl, (int, float)):
            lenscls = np.ones(x_train.shape[1]) * self.bounded_parameter(
                lenscl_bnds[0], lenscl_bnds[1], self.lenscl
            )
        if self.outputscl is not None:
            tau = self.bounded_parameter(var_bnds[0], var_bnds[1], self.outputscl)

        # On the 1st iteration, use initial guesses initialized to 1
        if retrain_count == 0:
            if self.lenscl is None:
                lengthscale_1 = self.bounded_parameter(
                    lenscl_bnds[0], lenscl_bnds[1], 1.0
                )
                lenscls = np.ones(x_train.shape[1]) * lengthscale_1
            if self.outputscl is None:
                tau = self.bounded_parameter(var_bnds[0], var_bnds[1], 1.0)
            white_var = self.bounded_parameter(
                white_var_bnds[0], white_var_bnds[1], 1.0
            )
        # On second iteration, base guesses on initial data values
        elif retrain_count == 1:
            if self.lenscl is None:
                initial_lenscls = np.array(
                    self.__set_lenscl_guess(lenscl_bnds[0], lenscl_bnds[1]),
                    dtype="float64",
                )
                lenscls = self.bounded_parameter(
                    lenscl_bnds[0], lenscl_bnds[1], initial_lenscls
                )
            if self.outputscl is None:
                initial_tau = np.array(
                    self.__set_outputscl(var_bnds[0], var_bnds[1]), dtype="float64"
                )
                tau = self.bounded_parameter(var_bnds[0], var_bnds[1], initial_tau)
            initial_white_var = np.array(
                self.__set_white_kern(white_var_bnds[0], white_var_bnds[1]),
                dtype="float64",
            )
            white_var = self.bounded_parameter(
                white_var_bnds[0], white_var_bnds[1], initial_white_var
            )
        # On all other iterations, use random guesses
        else:
            if self.lenscl is None:
                initial_lenscls = np.array(
                    rng.uniform(0.1, 100.0, x_train.shape[1]), dtype="float64"
                )
                lenscls = self.bounded_parameter(
                    lenscl_bnds[0], lenscl_bnds[1], initial_lenscls
                )
            if self.outputscl is None:
                tau = self.bounded_parameter(
                    var_bnds[0],
                    var_bnds[1],
                    np.array(rng.lognormal(0.0, 1.0), dtype="float64"),
                )
            white_var = self.bounded_parameter(
                white_var_bnds[0],
                white_var_bnds[1],
                np.array(rng.uniform(0.05, 10), dtype="float64"),
            )
        # try:
        #     print(lenscls.numpy(), tau.numpy(), white_var.numpy())
        # except:
        #     print(lenscls, tau, white_var)
        return lenscls, tau, white_var

    def set_gp_model(self, retrain_count):
        """
        Generates the GP model for the process in sklearn

        Parameters
        ----------
        retrain_count: int
            The number of times the GP will be (re)trained

        Returns
        --------
        gp_model: gpflow.models.GPR
            The untrained GP model with all hyperparameters set

        Raises
        ------
        AssertionError
            If retrains are not an integer greater than or equal to 0
        """

        assert (
            isinstance(retrain_count, int) and retrain_count >= 0
        ), "retrain_count must be an int greater than or equal to 0"
        data = self.set_gp_model_data()
        lenscls, tau, white_var = self.__init_hyper_parameters(retrain_count)

        if self.kernel.value == 3:
            gpKernel = gpflow.kernels.SquaredExponential(
                variance=tau, lengthscales=lenscls
            )
        elif self.kernel.value == 2:
            gpKernel = gpflow.kernels.Matern32(variance=tau, lengthscales=lenscls)
        else:
            gpKernel = gpflow.kernels.Matern52(variance=tau, lengthscales=lenscls)
        # Add White kernel
        gpKernel = gpKernel + gpflow.kernels.White(variance=white_var)

        # Build GP model
        gp_model = gpflow.models.GPR(data, kernel=gpKernel, noise_variance=10**-5)
        # condition_number = np.linalg.cond(model.kernel(X_Train))
        # Select whether the likelihood variance is trained
        gpflow.utilities.set_trainable(gp_model.likelihood.variance, False)
        if isinstance(self.lenscl, np.ndarray) or isinstance(self.lenscl, (float, int)):
            gpflow.utilities.set_trainable(
                gp_model.kernel.kernels[0].lengthscales, False
            )
        if self.outputscl is not None:
            gpflow.utilities.set_trainable(gp_model.kernel.kernels[0].variance, False)

        # print(gpflow.utilities.print_summary(gp_model))

        return gp_model

    def train_gp(self):
        """
        Trains the GP with restarts given training data.

        Raises
        ------
        AssertionError
            If self.feature_train_data is not an np.ndarray or is undefined

        Notes
        ------
        Sets the following parameters of self
        self.trained_hyperparams: list, the trained hyperparameters of the GP model
        self.fit_gp_model:  gpflow.models.GPR, the trained GP model
        self.posterior:  gpflow.mean_field.KFGaussian, the posterior of the GP model
        """
        assert isinstance(
            self.feature_train_data, np.ndarray
        ), "self.feature_train_data must be np.ndarray"
        assert (
            self.feature_train_data is not None
        ), "Must have training data. Run set_train_test_data() to generate"

        # Train the model multiple times and keep track of the model with the lowest minimum training loss
        best_minimum_loss = float("inf")
        best_model = None

        # If we are not retraining the GP, set the model once with default/set hyperparameters
        if self.retrain_GP == 0:
            best_model = self.set_gp_model(0)
        # Otherwise train the model and keep the best model over all retrains
        else:
            # While you still have retrains left
            for i in range(self.retrain_GP):
                # Create and fit the model
                gp_model = self.set_gp_model(i)
                # Build optimizer
                optimizer = gpflow.optimizers.Scipy()
                # Fit GP to training data
                aux = optimizer.minimize(
                    gp_model.training_loss,
                    gp_model.trainable_variables,
                    options={"maxiter": 10**9},
                    method="L-BFGS-B",
                )
                training_loss = gp_model.training_loss().numpy()
                if i == 0:
                    first_model = gp_model
                    first_loss = training_loss
                if aux.success:
                    # Check if this model has the best minimum training loss
                    if training_loss < best_minimum_loss:
                        best_minimum_loss = training_loss
                        best_model = gp_model

            # If we have no good models, use the first one
            if best_model is None:
                best_model = first_model
                best_minimum_loss = first_loss

        # Pull out kernel parameters after GP training
        outputscl_final = float(best_model.kernel.kernels[0].variance.numpy())
        lenscl_final = best_model.kernel.kernels[0].lengthscales.numpy()
        noise_final = float(best_model.kernel.kernels[1].variance.numpy())

        # Put hyperparameters in a list
        trained_hyperparams = [lenscl_final, noise_final, outputscl_final]

        # Assign self parameters
        self.trained_hyperparams = trained_hyperparams
        self.fit_gp_model = best_model
        self.posterior = self.fit_gp_model.posterior()

        # gpflow.utilities.print_summary(best_model)

    def __eval_gp_mean_var(self, data):
        """
        Calculates the GP mean and variance for a given input set and adds it to the instance of the data class

        Parameters
        -----------
        data: Data
            Data to evaluate GP for containing at least parameter sets (data.theta_vals) and state points (data.x_vals)

        Returns
        -------
        gp_mean: np.ndarray
            GP mean prediction for the data set
        gp_var: np.ndarray
            GP variance prediction for the data set
        gp_covar: np.ndarray
            GP covariance prediction for the data set

        """
        # Get data in vector form into array form
        if len(data.shape) < 2:
            data.reshape(1, -1)
        # scale eval_point if necessary
        if self.normalize == True:
            eval_points = self.scalerX.transform(data)
        else:
            eval_points = data

        eval_points_tf = tf.convert_to_tensor(eval_points)

        # with tf.GradientTape(persistent=True) as tape:
        #     # By default, only Variables are watched. For gradients with respect to tensors,
        #     # we need to explicitly watch them:
        #     tape.watch(eval_points_tf)
        #     # Evaluate GP given parameter set theta and state point value
        #     gp_mean_scl, gp_covar_scl = self.posterior.predict_f(
        #         eval_points_tf, full_cov=True
        #     )
        # Evaluate GP given parameter set theta and state point value
        gp_mean_scl, gp_covar_scl = self.posterior.predict_f(
                eval_points_tf, full_cov=True
            )
        # grad_mean_scl = tape.gradient(gp_mean_scl, eval_points_tf).numpy()

        # Remove dimensions of 1
        gp_mean_scl = gp_mean_scl.numpy()
        gp_covar_scl = np.squeeze(gp_covar_scl.numpy(), axis=0)

        # Unscale gp_mean and gp_covariance
        if self.normalize == True:
            gp_mean = self.scalerY.inverse_transform(
                gp_mean_scl.reshape(-1, 1)
            ).flatten()
            gp_covar = float(self.scalerY.scale_**2) * gp_covar_scl
            # grad_mean = self.scalerY.inverse_transform(
            #     grad_mean_scl.reshape(-1, 1)
            # ).flatten()
        else:
            gp_mean = gp_mean_scl
            gp_covar = gp_covar_scl
            # grad_mean = grad_mean_scl

        gp_var = np.diag(gp_covar)

        return gp_mean, gp_var, gp_covar

    def eval_gp_mean_var_misc(self, misc_data, featurized_misc_data, covar=False):
        """
        Evaluate the GP mean and variance for a heat map set

        Parameters
        -----------
        misc_data: Data
            Data to evaluate gp mean and variance for. Must contain data.theta_vals and data.x_vals
        featurized_misc_data: np.ndarray
            Featurized data to evaluate. Hint: Run featurize_data() to generate
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        -------
        misc_gp_mean: np.ndarray
            GP mean prediction for the data set
        misc_var_return: np.ndarray
            GP (co)variance prediction for the data set

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value

        Notes
        ------
        Populates misc_data.gp_mean and misc_data.gp_var with the GP mean and variance
        Also calculates the gp covariance matrix for the misc. data as a class object stored in misc_data.gp_covar
        """

        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(
            featurized_misc_data, np.ndarray
        ), "featurized_misc_data must be np.ndarray"
        assert len(featurized_misc_data) > 0, "Must have data"
        assert isinstance(covar, bool), "covar must be bool!"

        # Evaluate heat map data for GP
        misc_gp_mean, misc_gp_var, misc_gp_covar = self.__eval_gp_mean_var(
            featurized_misc_data
        )

        # Set data parameters
        misc_data.gp_mean = misc_gp_mean
        misc_data.gp_var = misc_gp_var
        misc_data.gp_covar = misc_gp_covar

        if covar == False:
            misc_var_return = misc_gp_var
        else:
            misc_var_return = misc_gp_covar

        return misc_gp_mean, misc_var_return

    def eval_gp_mean_var_test(self, covar=False):
        """
        Evaluate the GP mean and variance for a heat map set

        Parameters
        -----------
        test_data: Data
            Data to evaluate gp mean and variance for. Must contain data.theta_vals and data.x_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        -------
        test_gp_mean: np.ndarray
            GP mean prediction for the test set
        test_var_return: np.ndarray
            GP (co)variance prediction for the test set

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value

        Notes
        ------
        Populates test_data.gp_mean and test_data.gp_var with the GP mean and variance
        Also calculates the gp covariance matrix for the misc. data as a class object stored in test_data.gp_covar
        """

        assert (
            self.feature_test_data is not None
        ), "Must have testing data. Run set_train_test_data() to generate"
        assert (
            len(self.feature_test_data) > 0
        ), "Must have testing data. Run set_train_test_data() to generate"
        assert isinstance(covar, bool), "covar must be bool!"
        # Evaluate test data for GP
        test_gp_mean, test_gp_var, test_gp_covar = self.__eval_gp_mean_var(
            self.feature_test_data
        )

        # Set data parameters
        self.test_data.gp_mean = test_gp_mean
        self.test_data.gp_var = test_gp_var
        self.test_data.gp_covar = test_gp_covar

        if covar == False:
            test_var_return = test_gp_var
        else:
            test_var_return = test_gp_covar

        return test_gp_mean, test_var_return

    def eval_gp_mean_var_val(self, covar=False):
        """
        Evaluate the GP mean and variance for a heat map set

        Parameters
        -----------
        val_data: Data
            Data to evaluate gp mean and variance for. Must contain data.theta_vals and data.x_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        -------
        val_gp_mean: np.ndarray
            GP mean prediction for the test set
        val_var_return: np.ndarray
            GP (co)variance prediction for the test set

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value

        Notes
        ------
        Populates val_data.gp_mean and val_data.gp_var with the GP mean and variance
        Also calculates the gp covariance matrix for the misc. data as a class object stored in val_data.gp_covar
        """

        assert (
            self.feature_val_data is not None
        ), "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(
            self.feature_val_data, np.ndarray
        ), "self.feature_val_data must by np.ndarray"
        assert (
            len(self.feature_val_data) > 0
        ), "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(covar, bool), "covar must be bool!"

        # Evaluate test data for GP
        val_gp_mean, val_gp_var, val_gp_covar = self.__eval_gp_mean_var(
            self.feature_val_data
        )

        # Set data parameters
        self.gp_val_data.gp_mean = val_gp_mean
        self.gp_val_data.gp_var = val_gp_var
        self.gp_val_data.gp_covar = val_gp_covar

        if covar == False:
            val_var_return = val_gp_var
        else:
            val_var_return = val_gp_covar

        return val_gp_mean, val_var_return

    def eval_gp_mean_var_cand(self, covar=False):
        """
        Evaluate the GP mean and variance for a heat map set

        Parameters
        -----------
        cand_data: Data
            Data to evaluate gp mean and variance for. Must contain data.theta_vals and data.x_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        -------
        cand_gp_mean: np.ndarray
            GP mean prediction for the test set
        cand_var_return: np.ndarray
            GP (co)variance prediction for the test set

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value

        Notes
        ------
        Populates cand_data.gp_mean and cand_data.gp_var with the GP mean and variance
        Also calculates the gp covariance matrix for the misc. data as a class object stored in cand_data.gp_covar
        """

        assert (
            self.feature_cand_data is not None
        ), "Must have validation data. Run set_train_test_data() to generate"
        assert (
            len(self.feature_cand_data) > 0
        ), "Must have validation data. Run set_train_test_data() to generate"
        assert isinstance(covar, bool), "covar must be bool!"
        # Evaluate test data for GP
        cand_gp_mean, cand_gp_var, cand_gp_covar = self.__eval_gp_mean_var(
            self.feature_cand_data
        )

        # Set data parameters
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
    __init__(*): Constructor method
    get_dim_gp_data(): Defines the total dimension of data used by the GP
    featurize_data(data): Collects the featues of the GP into ndarray form from an instance of the Data class
    set_train_test_data(sep_fact, seed): Finds the simulation data to use as training/testing data
    __eval_gp_sse_var(data, covar): Calculates the GP mean and variance for a given input set
    eval_gp_sse_var_misc(misc_data, covar): Evaluate the GP mean and variance for miscellaneous data
    eval_gp_sse_var_test(covar): Evaluate the GP mean and variance for testing data
    eval_gp_sse_var_val(covar): Evaluate the GP mean and variance for validation data
    eval_gp_sse_var_cand(covar): Evaluate the GP mean and variance for candidate data
    calc_best_error(): Calculates the best error metrics for the GP
    __eval_gp_ei(sim_data, exp_data, ep_bias, best_error_metrics): Evaluates the expected improvement for the GP
    eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics): Evaluates the expected improvement for miscellaneous data
    eval_ei_test(exp_data, ep_bias, best_error_metrics): Evaluates the expected improvement for testing data
    eval_ei_val(exp_data, ep_bias, best_error_metrics): Evaluates the expected improvement for validation data
    eval_ei_cand(exp_data, ep_bias, best_error_metrics): Evaluates the expected improvement for candidate data
    add_next_theta_to_train_data(theta_best_sse_data): Adds the next parameter set to the training data
    """

    # Class variables and attributes

    def __init__(
        self,
        gp_sim_data,
        gp_val_data,
        cand_data,
        train_data,
        test_data,
        kernel,
        lenscl,
        noise_std,
        outputscl,
        retrain_GP,
        set_seed,
        normalize,
        feature_train_data,
        feature_test_data,
        feature_val_data,
        feature_cand_data,
    ):
        """
        Parameters
        ----------
        gp_sim_data: Data
            All simulation data for the GP
        gp_val_data: Data
            The validation data for the GP. None if not saving validation data
        cand_data: Data
            Candidate theta value for evaluation with GPBO_Driver.opt_with_scipy()
        train_data: Data
            The training data for the GP
        test_data: Data
            The testing data for the GP
        kernel: Kernel_enum
            Determines which GP Kerenel to use
        lenscl: float or None
            Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int
            The standard deviation of the noise
        outputscl: float or None
            Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_GP: int
            Number of times to (re)train the GP. If 0, the GP is not trained and default/initial hyperparameters are used
        set_seed: int or None
            Random seed
        normalize: bool
            Determines whether data is standardized (with sklearn RobustScaler) before training
        feature_train_data: np.ndarray
            The feature data for the training data in ndarray form
        feature_test_data: np.ndarray
            The feature data for the testing data in ndarray form
        feature_val_data: np.ndarray
            The feature data for the validation data in ndarray form
        feature_cand_data: np.ndarray
            The feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        # Constructor method
        # Inherit objects from GP_Emulator Base Class
        super().__init__(
            gp_sim_data,
            gp_val_data,
            cand_data,
            kernel,
            lenscl,
            noise_std,
            outputscl,
            retrain_GP,
            set_seed,
            normalize,
            feature_train_data,
            feature_test_data,
            feature_val_data,
            feature_cand_data,
        )
        # Add training and testing data as child features
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_init = (
            None  # Will be populated with the 1st instance of train data
        )

    def get_dim_gp_data(self):
        """
        Defines the total dimension of the input data used by the GP

        Returns
        -------
        dim_gp_data: int
            The dimensions of the input data that the GP will use
        """
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "self.gp_sim_data.theta_vals must exist!"

        # Just use number of theta dimensions for Type 1
        dim_gp_data = self.gp_sim_data.get_dim_theta()

        return dim_gp_data

    def featurize_data(self, data):
        """
        Collects the features (parameter set values) of the GP into ndarray form from an instance of the Data class

        Parameters
        -----------
        data: Data
            Data to evaluate GP for containing at least parameter sets (data.theta_vals)

        Returns
        -------
        feature_eval_data: np.ndarray
            The feature data for the GP

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or not defined

        """
        assert isinstance(data, Data), "data must be an instance of Data"
        assert np.all(
            data.theta_vals is not None
        ), "Must have validation data theta_vals and x_vals to evaluate the GP"

        # Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = data.theta_vals

        return feature_eval_data

    def set_train_test_data(self, sep_fact, shuffle_seed = None):
        """
        Splits simulated data into training and testing data

        Parameters
        ----------
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None
            Determines seed for randomizations. None if seed is random

        Returns
        -------
        train_data: Data
            Contains all input/output data and bounds for training data
        test_data: Data
            Contains all input/output data and bounds for testing data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Sets self.train_data, self.test_data, self.feature_train_data, self.feature_test_data, and self.feature_val_data
        """
        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data"
        assert np.all(
            self.gp_sim_data.x_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.y_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.bounds_x is not None
        ), "Must have simulation x bounds to create train/test data"
        assert np.all(
            self.gp_sim_data.bounds_theta is not None
        ), "Must have simulation theta bounds to create train/test data"

        # Get train test idx
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split(shuffle_seed)

        # Get train data and set it as an instance of the data class
        theta_train = self.gp_sim_data.theta_vals[train_idx]
        x_train = (
            self.gp_sim_data.x_vals
        )  # x_vals for Type 1 is the same as exp_data. No need to index x
        y_train = self.gp_sim_data.y_vals[train_idx]
        train_data = Data(
            theta_train,
            x_train,
            y_train,
            None,
            None,
            None,
            None,
            None,
            self.gp_sim_data.bounds_theta,
            self.gp_sim_data.bounds_x,
            sep_fact,
        )
        self.train_data = train_data

        # Get test data and set it as an instance of the data class
        theta_test = self.gp_sim_data.theta_vals[test_idx]
        x_test = (
            self.gp_sim_data.x_vals
        )  # x_vals for Type 1 is the same as exp_data. No need to index x
        y_test = self.gp_sim_data.y_vals[test_idx]
        test_data = Data(
            theta_test,
            x_test,
            y_test,
            None,
            None,
            None,
            None,
            None,
            self.gp_sim_data.bounds_theta,
            self.gp_sim_data.bounds_x,
            sep_fact,
        )
        self.test_data = test_data

        # Set training and validation data features in GP_Emulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)

        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data

        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data

        # Set the initial training data for the GP Emulator upon creation
        if self.train_data_init is None:
            self.train_data_init = feature_train_data

        return train_data, test_data

    def __eval_gp_sse_var(self, data, covar=False):
        """
        Evaluates GP model sse and sse variance and for standard GPBO for the data

        Parameters
        ----------
        data: Data
            Parameter sets you want to evaluate the sse and sse variance for
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        data.sse: np.ndarray
            The sse derived from gp_mean evaluated over the data
        var_return: np.ndarray
            The sse (co)variance derived from the GP model's (co)variance evaluated over the data

        Raises
        ------
        AssertionError
            If covar is not a boolean

        Notes
        ------
        Populates data.sse, data.sse_var, and data.sse_covar with the GP mean, variance, and covariance (standard GPBO emulates objecive function)
        """
        assert isinstance(covar, bool), "covar must be bool!"
        # For type 1, sse is the gp_mean
        data.sse = data.gp_mean
        data.sse_var = data.gp_var
        data.sse_covar = data.gp_covar

        if covar == False:
            var_return = data.sse_var
        else:
            var_return = data.sse_covar

        return data.sse, var_return

    def eval_gp_sse_var_misc(self, misc_data, covar=False):
        """
        Evaluates GP model sse and sse variance and for standard GPBO for miscellaneous data

        Parameters
        ----------
        misc_data: Data
            Parameter sets you want to evaluate the sse and sse variance for
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        misc_sse_mean: np.ndarray
            The sse derived from gp_mean evaluated over the data
        misc_sse_var: np.ndarray
            The sse (co)variance derived from the GP model's (co)variance evaluated over the data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        Populates data.sse, data.sse_var, and data.sse_covar with the GP mean, variance, and covariance (standard GPBO emulates objecive function)
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert np.all(
            misc_data.gp_mean is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(
            misc_data.gp_var is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_misc()"

        # For type 1, sse is the gp_mean
        misc_sse_mean, misc_sse_var = self.__eval_gp_sse_var(misc_data, covar)

        return misc_sse_mean, misc_sse_var

    def eval_gp_sse_var_test(self, covar=False):
        """
        Evaluates GP model sse and sse variance and for standard GPBO for test data

        Parameters
        ----------
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        test_sse_mean: np.ndarray
            The sse derived from gp_mean evaluated over the data
        test_sse_var: np.ndarray
            The sse (co)variance derived from the GP model's (co)variance evaluated over the data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        Populates data.sse, data.sse_var, and data.sse_covar with the GP mean, variance, and covariance (standard GPBO emulates objecive function)
        """
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert np.all(
            self.test_data.gp_mean is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"
        assert np.all(
            self.test_data.gp_var is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_test()"

        # For type 1, sse is the gp_mean
        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(self.test_data, covar)

        return test_sse_mean, test_sse_var

    def eval_gp_sse_var_val(self, covar=False):
        """
        Evaluates GP model sse and sse variance and for standard GPBO for validation data

        Parameters
        ----------
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        val_sse_mean: np.ndarray
            The sse derived from gp_mean evaluated over the data
        val_sse_var: np.ndarray
            The sse (co)variance derived from the GP model's (co)variance evaluated over the data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        Populates data.sse, data.sse_var, and data.sse_covar with the GP mean, variance, and covariance (standard GPBO emulates objecive function)
        """
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert np.all(
            self.gp_val_data.gp_mean is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(
            self.gp_val_data.gp_var is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"

        # For type 1, sse is the gp_mean
        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(self.gp_val_data, covar)

        return val_sse_mean, val_sse_var

    def eval_gp_sse_var_cand(self, covar=False):
        """
        Evaluates GP model sse and sse variance and for standard GPBO for candidate data

        Parameters
        ----------
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        cand_sse_mean: np.ndarray
            The sse derived from gp_mean evaluated over the data
        cand_sse_var: np.ndarray
            The sse (co)variance derived from the GP model's (co)variance evaluated over the data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        Populates data.sse, data.sse_var, and data.sse_covar with the GP mean, variance, and covariance (standard GPBO emulates objecive function)
        """
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert np.all(
            self.cand_data.gp_mean is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"
        assert np.all(
            self.cand_data.gp_var is not None
        ), "Must have the GP's mean and standard deviation. Hint: Use eval_gp_mean_var_val()"

        # For type 1, sse is the gp_mean
        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(self.cand_data, covar)

        return cand_sse_mean, cand_sse_var

    def calc_best_error(self):
        """
        Calculates the best error of the model

        Returns
        -------
        best_error: float
            The best error of the method
        be_theta: np.ndarray
            The parameter set associated with the best error of the method
        train_idc: int
            The index of the best error in the training data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        """
        assert self.train_data is not None, "self.train_data must exist!"
        assert isinstance(self.train_data, Data), "self.train_data must be type Data"
        assert np.all(
            self.train_data.y_vals is not None
        ), "self.train_data.y_vals and self.train_data.theta_vals must exist!"
        assert np.all(
            self.train_data.theta_vals is not None
        ), "self.train_data.y_vals and self.train_data.theta_vals must exist!"

        # Best error is the minimum sse value of the training data for Type 1
        best_error = np.min(self.train_data.y_vals)
        train_idc = np.argmin(self.train_data.y_vals)
        be_theta = self.train_data.theta_vals[train_idc]

        return best_error, be_theta, train_idc

    def __eval_gp_ei(self, sim_data, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates gp acquisition function. In this case, ei

        Parmaeters
        ----------
        sim_data: Data
            Simulated data to evaluate ei for
        exp_data: Data
            Experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint use calc_best_error()

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame
            pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Notes
        -----
        This function also sets sim_data.acq to the expected improvement
        """
        # Call instance of expected improvement class
        ei_class = Expected_Improvement(
            ep_bias,
            sim_data.gp_mean,
            sim_data.gp_covar,
            exp_data,
            best_error_metrics,
            self.seed,
            None,
        )
        # Call correct method of ei calculation
        ei, ei_terms_df = ei_class.type_1()
        # Add ei data to validation data class
        sim_data.acq = ei

        return ei, ei_terms_df

    def eval_ei_misc(self, misc_data, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates GPBO acquisition function (EI) for miscellaneous parameter sets

        Parmaeters
        ----------
        misc_data: Data
            Data to evaluate ei for
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and  best_error_x values of the method. Hint: use calc_best_error()

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(
            misc_data, exp_data, ep_bias, best_error_metrics
        )
        return ei, ei_terms_df

    def eval_ei_test(self, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates GPBO acquisition function (EI) for testing parameter sets

        Parmaeters
        ----------
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and  best_error_x values of the method. Hint: use calc_best_error()

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in test_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(
            self.test_data, exp_data, ep_bias, best_error_metrics
        )
        return ei, ei_terms_df

    def eval_ei_val(self, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates GPBO acquisition function (EI) for validation parameter sets

        Parmaeters
        ----------
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and  best_error_x values of the method. Hint: use calc_best_error()

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in gp_val_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(
            self.gp_val_data, exp_data, ep_bias, best_error_metrics
        )

        return ei, ei_terms_df

    def eval_ei_cand(self, exp_data, ep_bias, best_error_metrics):
        """
        Evaluates GPBO acquisition function (EI) for candidate parameter sets

        Parmaeters
        ----------
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and  best_error_x values of the method. Hint: use calc_best_error()

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in gp_val_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"
        ei, ei_terms_df = self.__eval_gp_ei(
            self.cand_data, exp_data, ep_bias, best_error_metrics
        )

        return ei, ei_terms_df

    def add_next_theta_to_train_data(self, theta_best_sse_data):
        """
        Adds parameter set which optimizes the acquisition function to the training data set

        Parameters
        ----------
        theta_best_sse_data: Data
            The class containing the data relavent to argmin(acq. func.) for a Type 1 (standard) GP

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        This function updates self.train_data.theta_vals, self.train_data.y_vals, and self.feature_train_data
        """
        assert self.train_data is not None, "self.train_data must be Data"
        assert isinstance(self.train_data, Data), "self.train_data must be Data"
        assert isinstance(theta_best_sse_data, Data), "theta_best_sse_data must be Data"
        assert all(
            isinstance(var, np.ndarray)
            for var in [self.train_data.theta_vals, self.train_data.y_vals]
        ), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"
        assert all(
            isinstance(var, np.ndarray)
            for var in [theta_best_sse_data.theta_vals, theta_best_sse_data.y_vals]
        ), "theta_best_sse_data.theta_vals and self.theta_best_sse_data.y_vals must be np.ndarray"
        # Update training theta, x, and y separately
        self.train_data.theta_vals = np.vstack(
            (self.train_data.theta_vals, theta_best_sse_data.theta_vals)
        )
        self.train_data.y_vals = np.concatenate(
            (self.train_data.y_vals, theta_best_sse_data.y_vals)
        )
        feature_train_data = self.featurize_data(self.train_data)

        # Reset training data feature array
        self.feature_train_data = feature_train_data


class Type_2_GP_Emulator(GP_Emulator):
    """
    The base class for Gaussian Processes
    Parameters

    Methods
    --------------
    __init__(*) : Constructor method
    get_dim_gp_data(): Defines the total dimension of input data used by the GP
    featurize_data(data): Collects the features of the GP into ndarray form from an instance of the Data class
    set_train_test_data(sep_fact, seed): Finds the simulation data to use as training/testing data
    __eval_gp_sse_var(data, method, exp_data, covar): Calculates the SSE mean and variance for a given input set
    eval_gp_sse_var_misc(misc_data, method, exp_data, covar): Evaluate the SSE mean and variance for miscellaneous data
    eval_gp_sse_var_test(method, exp_data, covar): Evaluate the SSE mean and variance for testing data
    eval_gp_sse_var_val(method, exp_data, covar): Evaluate the SSE mean and variance for validation data
    eval_gp_sse_var_cand(method, exp_data, covar): Evaluate the SSE mean and variance for candidate data
    calc_best_error(method, exp_data): Calculates the best error metrics for the GP
    __eval_gp_ei(sim_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples): Evaluates the expected improvement for the GP
    eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples): Evaluates the expected improvement for miscellaneous data
    eval_ei_test(exp_data, ep_bias, best_error_metrics, method, sg_mc_samples): Evaluates the expected improvement for testing data
    eval_ei_val(exp_data, ep_bias, best_error_metrics, method, sg_mc_samples): Evaluates the expected improvement for validation data
    eval_ei_cand(exp_data, ep_bias, best_error_metrics, method, sg_mc_samples): Evaluates the expected improvement for candidate data
    add_next_theta_to_train_data(theta_best_data): Adds the next parameter set to the training data
    """

    # Class variables and attributes
    def __init__(
        self,
        gp_sim_data,
        gp_val_data,
        cand_data,
        train_data,
        test_data,
        kernel,
        lenscl,
        noise_std,
        outputscl,
        retrain_GP,
        set_seed,
        normalize,
        feature_train_data,
        feature_test_data,
        feature_val_data,
        feature_cand_data,
    ):
        """
        Parameters
        ----------
        gp_sim_data: Data,
            All simulation data for the GP
        gp_val_data: Data
            The validation data for the GP. None if not saving validation data
        cand_data: Data
            Candidate theta value for evaluation with GPBO_Driver.opt_with_scipy()
        train_data: Data
            The training data for the GP
        testing_data: Data
            The testing data for the GP
        kernel: Kernel_enum
            Determines which GP Kerenel to use
        lenscl: float or None
            Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        noise_std: float, int
            The standard deviation of the noise
        outputscl: float or None
            Determines value of outputscale - None if hyperparameters will be updated during training
        retrain_GP: int
            Number of times to (re)train the GP. If 0, the GP is not trained and default/initial hyperparameters are used
        set_seed: int or None
            Random seed
        normalize: bool
            Determines whether data is standardized (with sklearn RobustScaler) before training
        feature_train_data: np.ndarray
            The feature data for the training data in ndarray form
        feature_test_data: np.ndarray
            The feature data for the testing data in ndarray form
        feature_val_data: np.ndarray
            The feature data for the validation data in ndarray form
        feature_cand_data: np.ndarray
            The feature data for the candidate theta data in ndarray. Used with GPBO_Driver.__opt_with_scipy()

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Constructor method
        # Inherit objects from GP_Emulator Base Class
        super().__init__(
            gp_sim_data,
            gp_val_data,
            cand_data,
            kernel,
            lenscl,
            noise_std,
            outputscl,
            retrain_GP,
            set_seed,
            normalize,
            feature_train_data,
            feature_test_data,
            feature_val_data,
            feature_cand_data,
        )
        # Set training and testing data as child class specific objects
        assert (
            isinstance(train_data, Data) or train_data is None
        ), "train_data must be instance of Data or None"
        assert (
            isinstance(test_data, Data) or train_data is None
        ), "test_data must be instance of Data or None"

        self.train_data = train_data
        self.test_data = test_data
        self.train_data_init = (
            None  # This will be populated with the first set of training thetas
        )

    def get_dim_gp_data(self):
        """
        Defines the total dimension of input data used by the GP

        Returns
        -------
        dim_gp_data: int
            Tthe cardinality of GP input data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data"
        assert np.all(
            self.gp_sim_data.x_vals is not None
        ), "self.gp_sim_data.x_vals and self.gp_sim_data.theta_vals must exist!"
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "self.gp_sim_data.x_vals and self.gp_sim_data.theta_vals must exist!"

        # Number of theta dimensions + number of x dimensions
        dim_gp_data = int(
            self.gp_sim_data.get_dim_x_vals() + self.gp_sim_data.get_dim_theta()
        )

        return dim_gp_data

    def featurize_data(self, data):
        """
        Collects the features of the GP into ndarray form from an instance of the Data class

        Parameters
        -----------
        data: Data
            Data to evaluate GP for containing at least data.theta_vals and data.x_vals

        Returns
        --------
        feature_eval_data: np.ndarray
            The feature data for the GP

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        """
        assert isinstance(data, Data), "data must be instance of Data"
        assert np.all(
            data.x_vals is not None
        ), "data.x_vals and data.theta_vals must exist!"
        assert np.all(
            data.theta_vals is not None
        ), "data.x_vals and data.theta_vals must exist!"

        # Assign feature evaluation data as theta and x values. Create empty list to store gp approximations
        feature_eval_data = np.concatenate((data.theta_vals, data.x_vals), axis=1)

        return feature_eval_data

    def set_train_test_data(self, sep_fact, shuffle_seed=None):
        """
        Splits the simulation data into GP training/testing data

        Parameters
        ----------
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        set_seed: int or None
            Determines seed for randomizations. None if seed is random

        Returns
        -------
        train_data: Data
            Contains all input/output data and bounds for GP training data
        test_data: Data
            Contains all input/output data and bounds for GP testing data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Sets self.train_data, self.test_data, self.feature_train_data, self.feature_test_data, and self.feature_val_data
        """
        assert isinstance(
            sep_fact, (float, int)
        ), "Separation factor must be float or int > 0"
        assert 0 < sep_fact <= 1, "sep_fact must be > 0 and less than or equal to 1!"
        assert isinstance(
            self.gp_sim_data, Data
        ), "self.gp_sim_data must be instance of Data"
        assert np.all(
            self.gp_sim_data.x_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.theta_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.y_vals is not None
        ), "Must have simulation x, theta, and y data to create train/test data"
        assert np.all(
            self.gp_sim_data.bounds_x is not None
        ), "Must have simulation x bounds to create train/test data"
        assert np.all(
            self.gp_sim_data.bounds_theta is not None
        ), "Must have simulation theta bounds to create train/test data"

        # Find train indeces
        train_idx, test_idx = self.gp_sim_data.train_test_idx_split(shuffle_seed)

        # Find unique theta_values
        unique_theta_vals = self.gp_sim_data.get_unique_theta()

        # Check which rows in theta_vals match the rows in Theta_unique based on theta_idx
        train_mask = np.isin(self.gp_sim_data.theta_vals, unique_theta_vals[train_idx])
        test_mask = np.isin(
            self.gp_sim_data.theta_vals, unique_theta_vals[train_idx], invert=True
        )

        # Get the indices of the matching rows
        train_rows_idx = np.all(train_mask, axis=1)
        test_rows_idx = np.all(test_mask, axis=1)

        # Use the indices to select the specific rows from theta_vals
        # Set training data and set it as an instance of the data class
        theta_train = self.gp_sim_data.theta_vals[train_rows_idx]
        x_train = self.gp_sim_data.x_vals[train_rows_idx]
        y_train = self.gp_sim_data.y_vals[train_rows_idx]
        train_data = Data(
            theta_train,
            x_train,
            y_train,
            None,
            None,
            None,
            None,
            None,
            self.gp_sim_data.bounds_theta,
            self.gp_sim_data.bounds_x,
            sep_fact,
        )
        self.train_data = train_data

        # Get test data and set it as an instance of the data class
        theta_test = self.gp_sim_data.theta_vals[test_rows_idx]
        x_test = self.gp_sim_data.x_vals[test_rows_idx]
        y_test = self.gp_sim_data.y_vals[test_rows_idx]
        test_data = Data(
            theta_test,
            x_test,
            y_test,
            None,
            None,
            None,
            None,
            None,
            self.gp_sim_data.bounds_theta,
            self.gp_sim_data.bounds_x,
            sep_fact,
        )
        self.test_data = test_data

        # Set training and validation data features in GP_Emulator base class
        feature_train_data = self.featurize_data(train_data)
        feature_test_data = self.featurize_data(test_data)

        self.feature_train_data = feature_train_data
        self.feature_test_data = feature_test_data

        if self.gp_val_data is not None:
            feature_val_data = self.featurize_data(self.gp_val_data)
            self.feature_val_data = feature_val_data

        # Set the initial training data for the GP Emulator upon creation
        if self.train_data_init is None:
            self.train_data_init = feature_train_data

        return train_data, test_data

    def __eval_gp_sse_var(self, data, method, exp_data, covar=False):
        """
        Evaluates GP model sse and sse (co)variance for emulator GPBO

        Parameters
        ----------
        data: Data
            Parameter sets you want to evaluate the sse and sse variance for
        method: GPBO_Methods
            Contains data for methods
        exp_data: Data
            The experimental data of the class. Must contain exp_data.x_vals and exp_data.y_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        sse_mean: tensor
            The sse derived from gp_mean evaluated over all state points
        var_return: tensor
            The sse (co)variance derived from the GP model's variance evaluated over all state points

        Raises
        ------
        AssertionError
            If covar is not a boolean

        Notes
        ------
        Also populates data.sse, data.sse_var, and data.sse_covar

        """
        assert isinstance(covar, bool), "covar must be bool!"

        # Find length of theta and number of unique x in data arrays
        len_theta = data.get_num_theta()
        len_x = len(data.get_unique_x())
        # Infer number of thetas
        num_uniq_theta = int(len_theta / len_x)

        # Reshape y_sim into n_theta rows x n_x columns
        indices = np.arange(0, len_theta, len_x)
        n_blocks = len(indices)
        # Slice y_sim into blocks of size len_x and calculate squared errors for each block
        gp_mean_resh = data.gp_mean.reshape(n_blocks, len_x)
        block_errors = gp_mean_resh - exp_data.y_vals[np.newaxis, :]
        residuals = block_errors.reshape(data.gp_covar.shape[0], -1)
        # Sum squared errors for each block
        sse_mean_org = np.sum((block_errors) ** 2, axis=1)
        sse_mean = sse_mean_org.flatten()

        # Calculate the sse variance. This SSE_variance CAN'T be negative
        sse_var_all = (
            2 * np.trace(data.gp_covar**2) + 4 * residuals.T @ data.gp_covar @ residuals
        )

        # Calculate individual variances Var(SSE[t1]), and Var(SSE[t2])
        if num_uniq_theta == 1:
            sse_var = sse_var_all
            sse_covar = sse_var
        else:
            sse_var = np.zeros(n_blocks)
            for i in range(n_blocks):
                # Get section of covariance matrix that corresponds to the covariance of the different thetas
                covar_t_t = data.gp_covar[
                    i * len_x : (i + 1) * len_x, i * len_x : (i + 1) * len_x
                ]
                # Get row of block error corresponding to this matrix
                res_theta = block_errors[i].reshape(-1, 1)
                # Calculate Variance
                sse_var[i] = (
                    2 * np.trace(covar_t_t**2) + 4 * res_theta.T @ covar_t_t @ res_theta
                )
            if num_uniq_theta == 2 and covar == True:
                sse_covar = sse_var_all
            else:
                sse_covar = None

        # Set class parameters
        data.sse = sse_mean
        data.sse_var = sse_var
        data.sse_covar = sse_covar

        if covar == False:
            var_return = data.sse_var
        else:
            var_return = data.sse_covar

        return sse_mean, var_return

    def eval_gp_sse_var_misc(self, misc_data, method, exp_data, covar=False):
        """
        Evaluates GP model sse and sse (co)variance for emulator GPBO for miscellaneous data

        Parameters
        ----------
        misc_data:
            Data parameter sets you want to evaluate the sse and sse variance for
        method: GPBO_Methods
            Contains data for methods
        exp_data: Data
            The experimental data of the class. Must contain exp_data.x_vals and exp_data.y_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        misc_sse_mean: tensor
            The sse derived from gp_mean evaluated over all state points
        misc_sse_var: tensor
            The sse (co)variance derived from the GP model's variance evaluated over all state points

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods class"
        assert all(
            isinstance(var, Data) for var in [misc_data, exp_data]
        ), "misc_data and exp_data must be type Data"
        assert np.all(
            misc_data.x_vals is not None
        ), "misc_data.x_vals and misc_data.theta_vals must exist!"
        assert np.all(
            misc_data.theta_vals is not None
        ), "misc_data.x_vals and misc_data.theta_vals must exist!"
        assert np.all(
            misc_data.gp_mean is not None
        ), "misc_data.gp_mean and misc_data.gp_var must exist. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(
            misc_data.gp_var is not None
        ), "misc_data.gp_mean and misc_data.gp_var must exist. Hint: Use eval_gp_mean_var_misc()"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"

        misc_sse_mean, misc_sse_var = self.__eval_gp_sse_var(
            misc_data, method, exp_data, covar
        )

        return misc_sse_mean, misc_sse_var

    def eval_gp_sse_var_test(self, method, exp_data, covar=False):
        """
        Evaluates GP model sse and sse (co)variance for emulator GPBO for GP testing data

        Parameters
        ----------
        method: GPBO_Methods
            Contains data for methods
        exp_data: Data
            The experimental data of the class. Must contain exp_data.x_vals and exp_data.y_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        test_sse_mean: tensor
            The sse derived from gp_mean evaluated over all state points
        test_sse_var: tensor
            The sse (co)variance derived from the GP model's variance evaluated over all state points
        """

        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods class"
        assert all(
            isinstance(var, Data) for var in [self.test_data, exp_data]
        ), "self.test_data and exp_data must be type Data"
        assert np.all(
            self.test_data.x_vals is not None
        ), "test_data.x_vals and test_data.theta_vals must exist!"
        assert np.all(
            self.test_data.theta_vals is not None
        ), "misc_data.x_vals and test_data.theta_vals must exist!"
        assert np.all(
            self.test_data.gp_mean is not None
        ), "test_data.gp_mean and test_data.gp_var must exist. Hint: Use eval_gp_mean_var_test()"
        assert np.all(
            self.test_data.gp_var is not None
        ), "test_data.gp_mean and test_data.gp_var must exist. Hint: Use eval_gp_mean_var_test()"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"

        test_sse_mean, test_sse_var = self.__eval_gp_sse_var(
            self.test_data, method, exp_data, covar
        )

        return test_sse_mean, test_sse_var

    def eval_gp_sse_var_val(self, method, exp_data, covar=False):
        """
        Evaluates GP model sse and sse (co)variance for emulator GPBO for GP validation data

        Parameters
        ----------
        method: GPBO_Methods
            Contains data for methods
        exp_data: Data
            The experimental data of the class. Must contain exp_data.x_vals and exp_data.y_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        val_sse_mean: tensor
            The sse derived from gp_mean evaluated over all state points
        val_sse_var: tensor
            The sse (co)variance derived from the GP model's variance evaluated over all state points

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods class"
        assert all(
            isinstance(var, Data) for var in [self.gp_val_data, exp_data]
        ), "self.gp_val_data and exp_data must be type Data"
        assert np.all(
            self.gp_val_data.x_vals is not None
        ), "gp_val_data.x_vals and gp_val_data.theta_vals must exist!"
        assert np.all(
            self.gp_val_data.theta_vals is not None
        ), "misc_data.x_vals and gp_val_data.theta_vals must exist!"
        assert np.all(
            self.gp_val_data.gp_mean is not None
        ), "gp_val_data.gp_mean and gp_val_data.gp_var must exist. Hint: Use eval_gp_mean_var_val()"
        assert np.all(
            self.gp_val_data.gp_var is not None
        ), "gp_val_data.gp_mean and gp_val_data.gp_var must exist. Hint: Use eval_gp_mean_var_val()"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"

        val_sse_mean, val_sse_var = self.__eval_gp_sse_var(
            self.gp_val_data, method, exp_data, covar
        )

        return val_sse_mean, val_sse_var

    def eval_gp_sse_var_cand(self, method, exp_data, covar=False):
        """
        Evaluates GP model sse and sse (co)variance for emulator GPBO for GP candidate parameter set data

        Parameters
        ----------
        method: GPBO_Methods
            Contains data for methods
        exp_data: Data
            The experimental data of the class. Must contain exp_data.x_vals and exp_data.y_vals
        covar: bool, default False
            Determines whether covariance (True) or variance (False) of sse is returned with the gp mean

        Returns
        --------
        cand_sse_mean: tensor
            The sse derived from gp_mean evaluated over all state points
        cand_sse_var: tensor
            The sse (co)variance derived from the GP model's variance evaluated over all state points
        """
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods class"
        assert all(
            isinstance(var, Data) for var in [self.cand_data, exp_data]
        ), "self.cand_data and exp_data must be type Data"
        assert np.all(
            self.cand_data.x_vals is not None
        ), "cand_data.x_vals and cand_data.theta_vals must exist!"
        assert np.all(
            self.cand_data.theta_vals is not None
        ), "misc_data.x_vals and cand_data.theta_vals must exist!"
        assert np.all(
            self.cand_data.gp_mean is not None
        ), "cand_data.gp_mean and cand_data.gp_var must exist. Hint: Use eval_gp_mean_var_val()"
        assert np.all(
            self.cand_data.gp_var is not None
        ), "cand_data.gp_mean and cand_data.gp_var must exist. Hint: Use eval_gp_mean_var_val()"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"

        cand_sse_mean, cand_sse_var = self.__eval_gp_sse_var(
            self.cand_data, method, exp_data, covar
        )

        return cand_sse_mean, cand_sse_var

    def calc_best_error(self, method, exp_data):
        """
        Calculates the best error of the model (sse) and squared error for each state point x (squared error)

        Parameters
        ----------
        method: GPBO_Methods
            Class containing method information
        exp_data: Data
            Experimental data. Must contain exp_data.x_vals, exp_data.theta_vals, and exp_data.y_vals

        Returns
        -------
        best_error: float
            The best error (sse) of the method
        be_theta: np.ndarray
            The parameter set associated with the best error value
        best_sq_error: np.ndarray
            Array of squared errors for each value of x
        org_train_idcs: list(int)
            The original training indices of be_theta

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods class"
        assert all(
            isinstance(var, Data) for var in [self.train_data, exp_data]
        ), "self.tain_data and exp_data must be type Data"
        assert np.all(
            self.train_data.x_vals is not None
        ), "self.train_data.x_vals, self.train_data.theta_vals, and self.train_data.y_vals must exist!"
        assert np.all(
            self.train_data.theta_vals is not None
        ), "self.train_data.x_vals, self.train_data.theta_vals, and self.train_data.y_vals must exist!"
        assert np.all(
            self.train_data.y_vals is not None
        ), "self.train_data.x_vals, self.train_data.theta_vals, and self.train_data.y_vals must exist!"
        assert np.all(
            exp_data.x_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"
        assert np.all(
            exp_data.y_vals is not None
        ), "exp_data.x_vals and exp_data.y_vals must exist!"

        # Find length of theta and x in data arrays
        len_theta = self.train_data.get_num_theta()
        len_x = len(self.train_data.get_unique_x())

        # #Reshape y_sim into n_theta rows x n_x columns
        indices = np.arange(0, len_theta, len_x)
        n_blocks = len(indices)
        # Slice y_sim into blocks of size len_x and calculate squared errors for each block
        train_y_resh = self.train_data.y_vals.reshape(n_blocks, len_x)
        ind_errors = (train_y_resh - exp_data.y_vals[np.newaxis, :]) ** 2

        # Sum squared errors for each block
        sse_vals = np.sum(ind_errors, axis=1)
        sse_train_vals = sse_vals.flatten()

        # List to array
        be_theta = self.train_data.theta_vals[int(np.argmin(sse_train_vals) * len_x)]
        org_train_idcs = [
            int(np.argmin(sse_train_vals) * len_x),
            int((np.argmin(sse_train_vals) + 1) * len_x),
        ]

        # Best error is the minimum of these values
        best_error = np.amin(sse_train_vals)
        best_sq_error = ind_errors[np.argmin(sse_vals)]

        return best_error, be_theta, best_sq_error, org_train_idcs

    def __eval_gp_ei(
        self,
        sim_data,
        exp_data,
        ep_bias,
        best_error_metrics,
        method,
        sg_mc_samples=2000,
    ):
        """
        Evaluates the (EI) acquisition function for a given data set

        Parmaeters
        ----------
        sim_data: Data
            Data to evaluate ei for
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias, The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            the best error (sse), best error parameter set, and best_error_x (squared error) values of the method. Hint: use calc_best_error()
        method: Method class
            Method for GP Emulation
        sg_mc_samples: int, default 2000
            Number of samples to use for the Tasmanian sparse grid or Monte Carlo approaches

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in sim_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Note:
        -----
        This function also sets sim_data.acq to the expected improvement values
        """
        assert (
            6 >= method.method_name.value >= 3
        ), "Must be using method 2A, 2B, 2C, or 2D"
        # Set sparse grid depth if applicable
        if method.sparse_grid == True or method.mc == True:
            assert (
                isinstance(sg_mc_samples, int) and sg_mc_samples > 0
            ), "sg_mc_samples must be positive int for sparse grid and Monte Carlo methods"
        # Call instance of expected improvement class
        ei_class = Expected_Improvement(
            ep_bias,
            sim_data.gp_mean,
            sim_data.gp_covar,
            exp_data,
            best_error_metrics,
            self.seed,
            sg_mc_samples,
        )
        # Call correct method of ei calculation
        ei, ei_terms_df = ei_class.type_2(method)
        # Add ei data to validation data class
        sim_data.acq = ei

        return ei, ei_terms_df

    def eval_ei_misc(
        self,
        misc_data,
        exp_data,
        ep_bias,
        best_error_metrics,
        method,
        sg_mc_samples=2000,
    ):
        """
        Evaluates the (EI) acquisition function for a miscellaneous data set

        Parmaeters
        ----------
        sim_data: Data
            Data to evaluate ei for
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error (sse), best error parameter set, and best_error_x (squared error) values of the method. Hint: use calc_best_error()
        method: Method class
            Method for GP Emulation
        sg_mc_samples: int, default 2000
            Number of samples to use for the Tasmanian sparse grid or Monte Carlo approaches

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in misc_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        This function will fail for the sparse grid and Monte Carlo methods when more than one parameter set is used since it requires a single sample covariance matrix
        """
        assert isinstance(misc_data, Data), "misc_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods"
        assert (
            6 >= method.method_name.value > 2
        ), "method must be Type 2. Hint: Must have method.method_name.value > 2"

        if method.method_name.value in [5, 6]:
            if len(misc_data.get_unique_theta()) > 1:
                raise ValueError(
                    "Sparse Grid and Monte Carlo methods require a single sample covariance matrix"
                )

        ei, ei_terms_df = self.__eval_gp_ei(
            misc_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples
        )

        return ei, ei_terms_df

    def eval_ei_test(
        self, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples=2000
    ):
        """
        Evaluates the (EI) acquisition function for a miscellaneous data set

        Parmaeters
        ----------
        sim_data: Data
            Data to evaluate ei for
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error (sse), best error parameter set, and best_error_x (squared error) values of the method. Hint: use calc_best_error()
        method: Method class
            Method for GP Emulation
        sg_mc_samples: int, default 2000
            Number of samples to use for the Tasmanian sparse grid or Monte Carlo approaches

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in misc_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        This function will fail for the sparse grid and Monte Carlo methods when more than one parameter set is used since it requires a single sample covariance matrix
        """
        assert isinstance(self.test_data, Data), "self.test_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "Error metric must be a tuple of length 3"
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods"
        assert (
            6 >= method.method_name.value > 2
        ), "method must be Type 2. Hint: Must have method.method_name.value > 2"
        if method.method_name.value in [5, 6]:
            if len(self.test_data.get_unique_theta()) > 1:
                raise ValueError(
                    "Sparse Grid and Monte Carlo methods require a single sample covariance matrix"
                )
        ei, ei_terms_df = self.__eval_gp_ei(
            self.test_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples
        )

        return ei, ei_terms_df

    def eval_ei_val(
        self, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples=2000
    ):
        """
        Evaluates gp acquisition function for validation data. In this case, ei

        Parmaeters
        ----------
        sim_data: Data
            Simualted data to evaluate ei for
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: Method class
            Method for GP Emulation
        sg_mc_samples: int, default 2000
            Number of to use for the Tasmanian sparse grid or MC approaches

        Returns
        -------
        ei: np.ndarray,
            The expected improvement of all the data in gp_val_data
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(self.gp_val_data, Data), "self.gp_val_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "best_error_metrics must be tuple of length 3"
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods"
        assert (
            6 >= method.method_name.value > 2
        ), "method must be Type 2. Hint: Must have method.method_name.value > 2"

        if method.method_name.value in [5, 6]:
            if len(self.gp_val_data.get_unique_theta()) > 1:
                raise ValueError(
                    "Sparse Grid and Monte Carlo methods require a single sample covariance matrix"
                )

        ei, ei_terms_df = self.__eval_gp_ei(
            self.gp_val_data,
            exp_data,
            ep_bias,
            best_error_metrics,
            method,
            sg_mc_samples,
        )

        return ei, ei_terms_df

    def eval_ei_cand(
        self, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples=2000
    ):
        """
        Evaluates gp acquisition function for the candidate theta data. In this case, ei

        Parmaeters
        ----------
        sim_data: Data
            Simualted data to evaluate ei for
        exp_data: Data
            The experimental data to evaluate ei with
        ep_bias: Exploration_Bias
            The exploration bias class
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error (sse), best error parameter set, and best_error_x (squared error) values of the method
        method: Method class
            Method for GP Emulation
        sg_mc_samples: int, default 2000
            Number of to use for the Tasmanian sparse grid or MC approaches

        Returns
        -------
        ei: np.ndarray
            The expected improvement of all the data in candidate feature
        ei_terms_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter sets

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(self.cand_data, Data), "self.cand_data must be type Data"
        assert isinstance(exp_data, Data), "exp_data must be type Data"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be type Exploration_bias"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "best_error_metrics must be tuple of length 3"
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods"
        assert (
            6 >= method.method_name.value > 2
        ), "method must be Type 2. Hint: Must have method.method_name.value > 2"

        if method.method_name.value in [5, 6]:
            if len(self.cand_data.get_unique_theta()) > 1:
                raise ValueError(
                    "Sparse Grid and Monte Carlo methods require a single sample covariance matrix"
                )

        ei, ei_terms_df = self.__eval_gp_ei(
            self.cand_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples
        )

        return ei, ei_terms_df

    def add_next_theta_to_train_data(self, theta_best_data):
        """
        Adds parameter set which optimizes the acquisition function to the training data set

        Parameters
        ----------
        theta_best_data: Data
            The class containing the data relavent to argmin(acq. func.) for a Type 1 (standard) GP

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        This function updates self.train_data.theta_vals, self.train_data.x_vals, self.train_data.y_vals, and self.feature_train_data
        """
        assert self.train_data is not None, "self.train_data must be Data"
        assert isinstance(self.train_data, Data), "self.train_data must be Data"
        assert isinstance(theta_best_data, Data), "theta_best_data must be Data"
        assert all(
            isinstance(var, np.ndarray)
            for var in [self.train_data.theta_vals, self.train_data.y_vals]
        ), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"
        assert all(
            isinstance(var, np.ndarray)
            for var in [theta_best_data.theta_vals, theta_best_data.y_vals]
        ), "self.train_data.theta_vals and self.train_data.y_vals must be np.ndarray"

        # Update training theta, x, and y separately
        self.train_data.theta_vals = np.vstack(
            (self.train_data.theta_vals, theta_best_data.theta_vals)
        )
        self.train_data.x_vals = np.vstack(
            (self.train_data.x_vals, theta_best_data.x_vals)
        )
        self.train_data.y_vals = np.concatenate(
            (self.train_data.y_vals, theta_best_data.y_vals)
        )
        feature_train_data = self.featurize_data(self.train_data)

        # Reset training data feature array
        self.feature_train_data = feature_train_data


##Again, composition instead of inheritance
class Expected_Improvement:
    """
    The base class for acquisition functions
    Parameters

    Methods
    --------------
    __init__(*): Constructor method
    __set_sg_def(dim): Sets the sparse grid depth
    __set_rand_vars(mean, covar): Sets random variables for MC integration
    type_1(): Calculates the expected improvement for Type 1 (standard) GPBO
    type_2(method): Calculates the expected improvement for Type 2 (emulator) GPBO
    __calc_ei_emulator(gp_mean, gp_var, y_target): Calculates the expected improvement for the independence approx.
    __calc_ei_log_emulator(gp_mean, gp_var, y_target): Calculates the expected improvement for the log independence approx.
    __ei_approx_ln_term(epsilon, gp_mean, gp_stdev, y_target): Calculates the integral for the log independence approx.
    __calc_ei_sparse(gp_mean, gp_var, y_target): Calculates the expected improvement for the sparse grid method
    __get_sparse_grids(dim, output=1,depth=10, rule="gauss-hermite-odd", verbose = False, alpha = 0): Gets the sparse grid
    __calc_ei_mc(gp_mean, gp_var, y_target): Calculates the expected improvement for the Monte Carlo method
    __bootstrap(pilot_sample, ns=100, alpha=0.05, seed = None): Bootstraps for the Monte Carlo method
    """

    def __init__(
        self,
        ep_bias,
        gp_mean,
        gp_covar,
        exp_data,
        best_error_metrics,
        set_seed,
        sg_mc_samples=2000,
    ):
        """
        Parameters
        ----------
        ep_bias: Exploration_Bias
            Class with information of exploration bias parameter
        gp_mean: tensor
            The GP model's mean
        gp_covar: tensor
            The GP model's covariance
        exp_data: Data
            The experimental data to evaluate ei with
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: use calc_best_error()
        set_seed: int or None
            Determines seed for randomizations. None if seed is random
        sg_mc_samples: int, default 2000
            The number of points to use for the Tasmanian sparse grid and Monte Carlo
        """
        assert len(gp_mean) == len(
            gp_covar
        ), "gp_mean and gp_covar must be arrays of the same length"
        assert len(gp_covar.shape) == 2, "gp_covar must be a 2D array"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "best_error_metrics must be a tuple of length 3"
        assert all(
            isinstance(arr, np.ndarray) for arr in (gp_mean, gp_covar, exp_data.y_vals)
        ), "gp_mean, gp_var, and exp_data.y_vals must be ndarrays"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be instance of Exploration_Bias"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(
            best_error_metrics[0], (float, int)
        ), "best_error_metrics[0] must be float or int. Calculate with GP_Emulator.calc_best_error()"
        assert isinstance(
            best_error_metrics[1], np.ndarray
        ), "best_error_metrics[1] must be np.ndarray"
        assert (
            isinstance(best_error_metrics[2], np.ndarray)
            or best_error_metrics[2] is None
        ), "best_error_metrics[2] must be np.ndarray (type 2 ei) or None (type 1 ei)"
        assert (
            isinstance(sg_mc_samples, int) or sg_mc_samples is None
        ), "sg_mc_samples must be int (MC and sparse grid) or None (other)"

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

        self.rng_rand = np.random.default_rng(self.seed)
        if self.seed is not None:
            self.rng_set = np.random.default_rng(self.seed)
        else:
            self.rng_set = self.rng_rand

    def __set_sg_def(self, dim):
        """
        Sets the sparse grid depth based on the maximum number of samples

        Parameters
        ----------
        dim: int
            The number of dimensions in the sparse grid

        Returns
        -------
        depth: int
            The depth of the sparse grid
        """
        depth = 0
        num_points = 0
        # Compute the maximum depth based on the budget
        while num_points <= self.samples_mc_sg:
            depth += 1
            # Generate the global grid with the current depth
            grid_p = Tasmanian.makeGlobalGrid(
                dim, 1, depth, "qphyperbolic", "gauss-hermite-odd"
            )

            # Get the number of points on the grid
            num_points = grid_p.getNumPoints()

            # Check if the number of points exceeds the budget
            if num_points > self.samples_mc_sg:
                if depth > 1:
                    depth -= 1
                break
        return depth

    def __set_rand_vars(self, mean=None, covar=None):
        """
        Sets random variables for MC integration

        Parameters
        ----------
        mean: np.ndarray or None, default None
            The mean of the random variables
        covar: np.ndarray or None, default None
            The covariance of the random variables

        Returns
        ---------
        random_vars: np.ndarray
            Array of multivariate normal random variables
        """
        rng = self.rng_set
        dim = len(self.exp_data.y_vals)
        mc_samples = self.samples_mc_sg  # Set 2000 MC samples

        eigvals, eigvecs = np.linalg.eigh(covar)

        # Get random standard variables
        random_vars_stand = rng.multivariate_normal(
            np.zeros(dim), np.eye(dim), mc_samples
        )
        # If we have a mean and a variance
        if mean is not None or covar is not None:
            # Use the mvn function directly to get the random variables if matrix is Positive Definite
            if np.all(eigvals > 1e-7):
                random_vars = rng.multivariate_normal(
                    mean, np.real(covar), mc_samples, tol=1e-5, method="eigh"
                )
            # Otherwise, use the LDL decomposition
            else:
                lu, d, perm = scipy.linalg.ldl(
                    np.real(covar), lower=True
                )  # Use the lower part
                sqrt_d = np.sqrt(np.diag(d))[:, np.newaxis]
                random_vars = (
                    mean[:, np.newaxis] + lu[:, perm] @ (sqrt_d * random_vars_stand.T)
                ).T
                np.save("mean_mc.npy", mean)
                np.save("covar_mc.npy", covar)

        return random_vars

    def type_1(self):
        """
        Calculates expected improvement of type 1 (standard) GPBO given gp_mean, gp_var, and best_error data

        Returns
        -------
        ei: np.ndarray
            The expected improvement of the parameter set
        ei_term_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        columns = ["best_error", "z", "cdf", "pdf", "ei_term_1", "ei_term_2", "ei"]
        ei_term_df = pd.DataFrame(columns=columns)

        ei = np.zeros(len(self.gp_mean))

        for i in range(len(self.gp_mean)):
            pred_stdev = np.sqrt(self.gp_var[i])  # 1xn_test
            # Checks that all standard deviations are positive
            if pred_stdev > 0:
                # Calculates z-score based on Eq. 6b in Wang and Dowling (2022), COCHE
                z = (
                    self.best_error * self.ep_bias.ep_curr - self.gp_mean[i]
                ) / pred_stdev  # scaler
                # Calculates ei based on Eq. 6a in Wang and Dowling (2022), COCHE
                # Explotation term
                ei_term_1 = (
                    self.best_error * self.ep_bias.ep_curr - self.gp_mean[i]
                ) * norm.cdf(
                    z
                )  # scaler
                # Exploration Term
                ei_term_2 = pred_stdev * norm.pdf(z)  # scaler
                ei[i] = ei_term_1 + ei_term_2  # scaler

                # Create a temporary DataFrame for the current row
                row_data = pd.DataFrame(
                    [
                        [
                            self.best_error,
                            z,
                            norm.cdf(z),
                            norm.pdf(z),
                            ei_term_1,
                            ei_term_2,
                            ei[0],
                        ]
                    ],
                    columns=columns,
                )

            else:
                # Sets ei to zero if standard deviation is zero
                ei[i] = 0
                # Create a temporary DataFrame for the current row
                row_data = pd.DataFrame(
                    [[self.best_error, None, None, None, None, None, ei]],
                    columns=columns,
                )

            # Concatenate the temporary DataFrame with the main DataFrame
            ei_term_df = pd.concat(
                [ei_term_df.astype(row_data.dtypes), row_data], ignore_index=True
            )
        return ei, ei_term_df

    def type_2(self, method):
        """
        Calculates expected improvement of type 2 (emulator) GPBO

        Parameters
        ----------
        method: GPBO_Methods
            Fully defined methods class which determines which method will be used

        Returns
        -------
        ei: np.ndarray
            The expected improvement of the parameter set
        ei_term_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        ei_term_df = pd.DataFrame()
        assert isinstance(
            self.best_error_x, np.ndarray
        ), "best_error_metrics[1] must be np.ndarray for type 2 ei calculations"
        assert isinstance(method, GPBO_Methods), "method must be type GPBO_Methods"
        # Num thetas = #gp mean pts/number of x_vals for Type 2
        num_thetas = int(len(self.gp_mean) / self.exp_data.get_num_x_vals())
        # Define n as the number of x values
        n = self.exp_data.get_num_x_vals()
        # Initialize array of eis for eacch theta
        ei = np.zeros(num_thetas)

        # Loop over number of thetas in theta_val_set
        for i in range(num_thetas):  # 1 ei per theta and also 1 sse per theta
            # Get gp mean and var for each set of x values
            # for ei, ensure that a gp mean and gp_var corresponding to a certain theta are sent
            gp_mean_i = self.gp_mean[i * n : (i + 1) * n]
            gp_var_i = self.gp_var[i * n : (i + 1) * n]

            # Calculate ei for a given theta (ei for all x over each theta)

            if method.method_name.value == 3:  # 2A
                # Calculate ei for a given theta (ei for all x over each theta)
                ei[i], row_data = self.__calc_ei_emulator(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            elif method.method_name.value == 4:  # 2B
                ei[i], row_data = self.__calc_ei_log_emulator(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            elif method.method_name.value == 5:  # 2C
                ei[i], row_data = self.__calc_ei_sparse(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            elif method.method_name.value == 6:  # 2D
                ei[i], row_data = self.__calc_ei_mc(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            else:
                raise ValueError(
                    "method.method_name.value must be 3 (2A), 4 (2B), 5 (2C), or 6 (2D)"
                )

        # Concatenate the temporary DataFrame with the main DataFrame
        ei_term_df = pd.concat([ei_term_df, row_data], ignore_index=True)
        ei_term_df.columns = row_data.columns.tolist()

        return ei, ei_term_df

    def __calc_ei_emulator(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with an independence approximation (2A)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points (x) for a given parameter set
        gp_variance: np.ndarray
            Model variance at state points (x) for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_temp: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        """
        # Create column names
        columns = [
            "bound_l",
            "bound_u",
            "cdf_l",
            "cdf_u",
            "eta_l",
            "eta_u",
            "psi_l",
            "psi_u",
            "ei_term1",
            "ei_term2",
            "ei_term3",
            "ei",
            "ei_total",
        ]

        # Initialize ei as all zeros
        ei = np.zeros(len(gp_var))
        # Create a mask for values where var > 0. Set a value of 1e-14?
        pos_stdev_mask = gp_var > 0

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_var_val = gp_var[valid_indices]
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            best_errors_x = self.best_error_x[valid_indices]

            # If variance is close to zero this is important
            with np.errstate(divide="warn"):
                # Creates upper and lower bounds and described by Equation X in Manuscript
                bound_a = (
                    (y_target_val - gp_mean_val)
                    + np.sqrt(best_errors_x * self.ep_bias.ep_curr)
                ) / pred_stdev_val
                bound_b = (
                    (y_target_val - gp_mean_val)
                    - np.sqrt(best_errors_x * self.ep_bias.ep_curr)
                ) / pred_stdev_val
                bound_lower = np.minimum(bound_a, bound_b)
                bound_upper = np.maximum(bound_a, bound_b)

                # Creates EI terms in terms of Equation X in Manuscript
                ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
                ei_term1_comp2 = (best_errors_x * self.ep_bias.ep_curr) - (
                    y_target_val - gp_mean_val
                ) ** 2

                ei_term2_comp1 = 2 * (y_target_val - gp_mean_val) * pred_stdev_val
                ei_eta_upper = -np.exp(-(bound_upper**2) / 2) / np.sqrt(2 * np.pi)
                ei_eta_lower = -np.exp(-(bound_lower**2) / 2) / np.sqrt(2 * np.pi)
                ei_term2_comp2 = ei_eta_upper - ei_eta_lower

                ei_term3_comp1 = bound_upper * ei_eta_upper
                ei_term3_comp2 = bound_lower * ei_eta_lower

                ei_term3_comp3 = (1 / 2) * scipy.special.erf(bound_upper / np.sqrt(2))
                ei_term3_comp4 = (1 / 2) * scipy.special.erf(bound_lower / np.sqrt(2))

                ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp3
                ei_term3_psi_lower = ei_term3_comp2 + ei_term3_comp4

                ei_term1 = ei_term1_comp1 * ei_term1_comp2
                ei_term2 = ei_term2_comp1 * ei_term2_comp2
                ei_term3 = -gp_var_val * (ei_term3_psi_upper - ei_term3_psi_lower)

                # Set EI values of indecies where pred_stdev > 0
                ei[valid_indices] = ei_term1 + ei_term2 + ei_term3

            # The Ei is the sum of the ei at each value of x
            ei_temp = np.sum(ei)
            row_data_lists = pd.DataFrame(
                [
                    [
                        bound_lower,
                        bound_upper,
                        norm.cdf(bound_lower),
                        norm.cdf(bound_upper),
                        ei_eta_lower,
                        ei_eta_upper,
                        ei_term3_psi_lower,
                        ei_term3_psi_upper,
                        ei_term1,
                        ei_term2,
                        ei_term3,
                        ei,
                        ei_temp,
                    ]
                ],
                columns=columns,
            )
        else:
            ei_temp = 0
            row_data_lists = pd.DataFrame(
                [
                    [
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        ei_temp,
                    ]
                ],
                columns=columns,
            )

        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_temp, row_data

    def __calc_ei_log_emulator(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a log-scaled independence approximation (2B)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points (x) for a given parameter set
        gp_variance: np.ndarray
            Model variance at state points (x) for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_temp: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        columns = [
            "best_error",
            "bound_l",
            "bound_u",
            "ei_term1",
            "ei_term2",
            "ei",
            "ei_total",
        ]

        # Initialize ei as all zeros
        ei = np.zeros(len(gp_var))

        # Create a mask for values where pred_stdev > 0
        pos_stdev_mask = gp_var > 0
        # best_errors_x_all = np.log(self.best_error_x)
        best_errors_x_all = np.log(np.where(self.best_error_x < 1e-16, 1e-16, self.best_error_x))

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            best_errors_x = copy.deepcopy(best_errors_x_all)[valid_indices]
            # Important when stdev is close to 0
            with np.errstate(divide="warn"):
                # Creates upper and lower bounds and described by Alex Dowling's Derivation
                bound_a = (
                    (y_target_val - gp_mean_val)
                    + np.sqrt(np.exp(best_errors_x * self.ep_bias.ep_curr))
                ) / pred_stdev_val  # 1xn
                bound_b = (
                    (y_target_val - gp_mean_val)
                    - np.sqrt(np.exp(best_errors_x * self.ep_bias.ep_curr))
                ) / pred_stdev_val  # 1xn
                bound_lower = np.minimum(bound_a, bound_b)
                bound_upper = np.maximum(bound_a, bound_b)

                # Calculate EI
                args = (gp_mean_val, pred_stdev_val, y_target_val, self.ep_bias.ep_curr)
                ei_term_1 = (best_errors_x * self.ep_bias.ep_curr) * (
                    norm.cdf(bound_upper) - norm.cdf(bound_lower)
                )
                ei_term_2_out = np.array(
                    [
                        integrate.quad(
                            self.__ei_approx_ln_term, bl, bu, args=(gm, ps, yt)
                        )
                        for bl, bu, gm, ps, yt in zip(
                            bound_lower,
                            bound_upper,
                            gp_mean_val,
                            pred_stdev_val,
                            y_target_val,
                        )
                    ]
                )

                ei_term_2 = (-2) * ei_term_2_out[:, 0]
                term_2_abs_err = ei_term_2_out[:, 1]

                # Add ei values to correct indecies.
                ei[valid_indices] = ei_term_1 + ei_term_2

            # The Ei is the sum of the ei at each value of x
            ei_temp = np.sum(ei)
            row_data_lists = pd.DataFrame(
                [
                    [
                        best_errors_x,
                        bound_lower,
                        bound_upper,
                        ei_term_1,
                        ei_term_2,
                        ei,
                        ei_temp,
                    ]
                ],
                columns=columns,
            )
        else:
            ei_temp = 0
            row_data_lists = pd.DataFrame(
                [[best_errors_x_all, "N/A", "N/A", "N/A", "N/A", "N/A", ei_temp]],
                columns=columns,
            )

        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_temp, row_data

    def __ei_approx_ln_term(self, epsilon, gp_mean, gp_stdev, y_target):
        """
        Calculates the integrand of expected improvement intregral for the log independence approximation

        Parameters
        ----------
        epsilon: float
            The random variable over which we integrate
        gp_mean: np.ndarray
            GP model mean
        gp_stdev: np.ndarray
            GP model stdev
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_term_2_integral: np.ndarray
            The expected improvement for term 2 of the GP model for method 2B
        """
        # Define inside term as the maximum of 1e-14 or abs((y_target - gp_mean - gp_stdev*epsilon))
        inside_term = max(1e-14, abs((y_target - gp_mean - gp_stdev * epsilon)))

        ei_term_2_integral = math.log(inside_term) * norm.pdf(epsilon)

        return ei_term_2_integral

    def __calc_ei_sparse(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a sparse grid approach (2C)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points (x) for a given parameter set
        gp_var: np.ndarray
            Model variance at state points (x) for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_temp: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        Notes
        -----
        To apply the sparse grid method on multiple parameter sets you must loop over each parameter set, calculate the posterior mean and variance, and then
        apply the sparse grid method to calculate EI for each parameter set.
        If the covariance matrix is not positive definite, the LDL decomposition is used instead of Cholesky factorization.
        """
        columns = ["best_error", "sse_temp", "improvement", "ei_total"]

        # Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0)
        pos_stdev_mask = gp_var >= 0

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            ndims = len(y_target)
            # Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            gp_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            gp_mean_min_y = y_target_val - gp_mean_val

            # #Obtain Sparse Grid points and weights
            # Get maximum depth given number of points p
            sg_depth = self.__set_sg_def(ndims)
            points_p, weights_p = self.__get_sparse_grids(
                ndims, output=1, depth=sg_depth, rule="gauss-hermite-odd", verbose=False
            )

            # Diagonalize covariance matrix
            try:
                # As long as the covariance matrix is positive definite use Cholesky decomposition
                L = scipy.linalg.cholesky(np.real(self.gp_covar), lower=True)
            except:
                # If it is not, use LDL decomposition instead
                lu, d, perm = scipy.linalg.ldl(
                    np.real(self.gp_covar), lower=True
                )  # Use the upper part
                L = lu[:, perm] @ np.diag(np.sqrt(d))
                np.save("covar_sg.npy", self.gp_covar)

            transformed_points = L @ points_p.T
            gp_random_vars = self.gp_mean[:, np.newaxis] + np.sqrt(2) * (
                transformed_points
            )
            sse_temp = np.sum((y_target[:, np.newaxis] - gp_random_vars) ** 2, axis=0)
            # Apply max operator (equivalent to max[(best_error*ep) - SSE_Temp,0])
            error_diff = self.best_error * self.ep_bias.ep_curr - sse_temp
            # Smooth max improvement function
            improvement = (0.5) * (error_diff + np.sqrt(error_diff**2 + 1e-7))

            # Calculate EI_temp using vectorized operations
            ei_temp = (np.pi ** (-ndims / 2)) * np.dot(weights_p, improvement)

        else:
            ei_temp = 0
            sse_temp = "N/A"
            improvement = "N/A"

        row_data_lists = pd.DataFrame(
            [[self.best_error, sse_temp, improvement, ei_temp]], columns=columns
        )
        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_temp, row_data

    def __get_sparse_grids(
        self, dim, output=1, depth=10, rule="gauss-hermite-odd", verbose=False, alpha=0
    ):
        """
        This function builds a sparse grid

        Parameters
        -----------
        dim: int
            Sparse grids dimension
        output: int, default 1
            Output level for function that would be interpolated
        depth: int, default 10
            Depth level. Controls density of abscissa points. Uses qphyperbolic level system
        rule: str, default 'gauss-hermite-odd'
            Quadrature rule
        verbose: bool, default False
            Determines Whether or not plot of sparse grid is shown
        alpha: int, default 0
            Specifies $\alpha$ parameter for the integration weight $\rho(x)$

        Returns
        --------
        points_p: np.ndarray
            The sparse grid points
        weights_p: np.ndarray
            The Gauss-Hermite Quadrature Rule Weights

        Notes
        ------
        A figure shows a 2D sparse grid if verbose = True
        """
        # Get grid points and weights
        grid_p = Tasmanian.makeGlobalGrid(dim, output, depth, "qphyperbolic", rule)
        points_p = grid_p.getPoints()
        weights_p = grid_p.getQuadratureWeights()
        if verbose == True:
            # If verbose is true print the sparse grid
            for i in range(len(points_p)):
                plt.scatter(points_p[i, 0], points_p[i, 1])
                plt.title("Sparse Grid of " + rule.title(), fontsize=20)
                plt.xlabel(r"$ϵ$ Dimension 1", fontsize=20)
                plt.ylabel(r"$ϵ$ Dimension 2", fontsize=20)
            plt.show()
        return points_p, weights_p

    def __calc_ei_mc(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a Monte Carlo approach (2D)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points x for a given parameter set
        gp_variance: np.ndarray
            Model variance at state points x for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_mean: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        Note
        -----
        To apply the Monte Carlo method on multiple parameter sets you must loop over each parameter set, calculate the posterior mean and variance, and then
        apply the MC method to calculate EI for each parameter set.
        """
        # Set column names
        columns = [
            "best_error",
            "sse_temp",
            "improvement",
            "ci_lower",
            "ci_upper",
            "ei_total",
        ]

        # Calc EI
        # Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0)
        pos_stdev_mask = gp_var >= 0

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # Set random variables for MC integration
            self.random_vars = self.__set_rand_vars(self.gp_mean, self.gp_covar)
            sse_temp = np.sum(
                (y_target[:, np.newaxis].T - self.random_vars) ** 2, axis=1
            )
            error_diff = self.best_error * self.ep_bias.ep_curr - sse_temp
            # Smooth max improvement function
            improvement = (0.5) * (
                error_diff + np.sqrt(error_diff**2 + 1e-7)
            ).reshape(-1, 1)
            # Flatten improvement
            ei_temp = improvement.flatten()

        else:
            ei_temp = 0
            sse_temp = "N/A"
            improvement = "N/A"

        # Calc monte carlo integrand for each theta and add it to the total
        ei_mean = np.average(ei_temp)  # y.sum()/len(y)
        # Note: Domain for random variable is 0-1, so V for MC is 1

        # Perform bootstrapping
        ci_interval = self.__bootstrap(ei_temp, ns=100, alpha=0.05)

        ci_l = ci_interval[0]
        ci_u = ci_interval[1]

        row_data_lists = pd.DataFrame(
            [[self.best_error, sse_temp, improvement, ci_l, ci_u, ei_temp]],
            columns=columns,
        )
        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_mean, row_data

    def __bootstrap(self, pilot_sample, ns=100, alpha=0.05):
        """
        Bootstrapping code for Monte Carlo method. Generously provided by Ryan Smith.

        Parameters
        ----------
        pilot_sample: np.ndarray (n_samples x dim param set)
            The samples to perform bootstrapping on
        ns: int, default 100
            Number of bootstrapping samples
        alpha: float, default 0.05
            On interval (0,1). The level of significance associated with the bootstrapping
        set_seed: int or None, default None
            Seed associated with bootstrapping

        Returns
        --------
        ci_percentile: np.ndarray
            The confidence interval of the MC samples
        """
        # pilot_sample has one column per rv, one row per observation
        # alpha is the level of significance; 0.05 for 95% confidence interval
        quantiles = np.array([alpha * 0.5, 1.0 - alpha * 0.5])

        # Determine mean of all original samples and its shape
        theta_orig = np.mean(pilot_sample, axis=0)

        # Initialize bootstrap samples as zeros
        theta_bs = np.zeros(tuple([ns] + list(theta_orig.shape)))

        # Create bootstrap samples
        for ibs in range(ns):
            samples = self.rng_set.choice(
                pilot_sample, size=pilot_sample.shape[0], replace=True
            )
            theta_bs[ibs, ...] = np.mean(samples, axis=0)

        # percentile CI
        ci_percentile = np.quantile(theta_bs, quantiles, 0)

        # return theta_orig, theta_bs, CI_percentile
        return ci_percentile


class Exploration_Bias:
    """
    Base class for methods of calculating explroation bias at each bo iter

    Methods
    -------
    __init__(*): Constructor method
    __bound_ep(ep_val): Bounds the value of a given exploration parameter between the minimum and maximum value
    set_ep(): Updates value of exploration parameter based on one of the four alpha heuristics
    __set_ep_constant(): Creates a value for the exploration parameter based off of a constant value
    __set_ep_decay(): Creates a value for the exploration parameter based off of a decay heuristic
    __set_ep_boyle(): Creates a value for the exploration parameter based off of a Boyle heuristic
    __set_ep_jasrasaria(): Creates a value for the exploration parameter based off of a Jasrasaria heuristic
    """

    def __init__(
        self,
        ep0,
        ep_curr,
        ep_enum,
        bo_iter,
        bo_iter_max,
        ep_inc,
        ep_f,
        improvement,
        best_error,
        mean_of_var,
    ):
        """
        Parameters
        ----------
        ep0: float
            The original exploration parameter value
        ep_curr: float
            The current exploration parameter value
        ep_enum: Enum
            Whether Boyle, Jasrasaria, Constant, or Decay ep method will be used
        bo_iter: int
            The number of the current BO iteration
        bo_iter_max: int
            The maximum number of BO iterations
        e_inc: float
            The increment for the Boyle's method for calculating exploration parameter: Recommendation is 1.5
        ep_f: float
            The final exploration parameter value: Recommendation is 0
        improvement: bool
            Determines whether last objective was an improvement
        best_error: float
            The lowest error objective value in the training data
        mean_of_var: float
            The value of the average of all posterior variances

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        For all methods, ep is on domain [0.5, 2] inclusive
        """
        assert all(
            (isinstance(param, (float, int)) or param is None)
            for param in [ep0, ep_curr, ep_inc, ep_f, best_error, mean_of_var]
        ), "ep0, ep_curr, ep_inc, ep_f, best_error, and mean_of_var must be int, float, or None"
        assert (
            isinstance(ep_enum, Enum) == True
        ), "ep_enum must be an Enum instance of Class Ep_enum"
        assert (
            isinstance(improvement, bool) == True or improvement is None
        ), "improvement must be bool or None"
        assert all(
            (isinstance(param, (int)) or param is None)
            for param in [bo_iter, bo_iter_max]
        ), "bo_iter and bo_iter_max must be int or None"
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
        # Set ep max and min based off of mathematical bound reasoning
        self.ep_max = 2
        self.ep_min = 0.5

    def __bound_ep(self, ep_val):
        """
        Bounds the value of a given exploration parameter between the minimum and maximum value

        Parameters
        ----------
        ep_val: int or float
            The value of the exploration parameter

        Returns
        --------
        ep_val: int or float
            The value of the exploration parameter within self.ep_min and self.ep_max
        """
        assert isinstance(ep_val, (float, int)), "ep_val must be float or int!"
        if ep_val > self.ep_max:
            warnings.warn("setting ep_val to self.ep_max because it was too large")
            ep_val = self.ep_max
        elif ep_val < self.ep_min:
            warnings.warn("setting ep_val to self.ep_min because it was too small")
            ep_val = self.ep_min
        else:
            assert (
                self.ep_max >= ep_val >= self.ep_min
            ), "Starting exploration bias (ep0) must be greater than or equal to 0.5!"

        return ep_val

    def set_ep(self):
        """
        Updates value of exploration parameter based on one of the four alpha heuristics

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        --------
        Sets the current exploration parameter self.ep_curr, but does not return anything. Use Exploration_Bias.ep_curr() to return it

        """
        # Set ep0 and ep_f to the max if they are too large
        if self.ep0 is not None:
            self.ep0 = self.__bound_ep(self.ep0)
        if self.ep_f is not None:
            self.ep_f = self.__bound_ep(self.ep_f)

        if self.ep_enum.value == 1:  # Constant if using constant method
            assert self.ep0 is not None
            ep = self.__set_ep_constant()

        elif self.ep_enum.value == 2:  # Decay
            assert self.ep0 is not None
            assert self.ep_f is not None
            assert self.bo_iter_max is not None
            ep = self.__set_ep_decay()

        elif self.ep_enum.value == 3:  # Boyle
            assert self.ep0 is not None
            assert self.ep_inc is not None
            ep = self.__set_ep_boyle()

        else:  # Jasrasaria
            ep = self.__set_ep_jasrasaria()

        # Set current ep to new ep
        self.ep_curr = ep

    def __set_ep_constant(self):
        """
        Creates a value for the exploration parameter based off of a constant value

        Returns
        --------
        ep: float
            The exploration parameter for the iteration
        """
        ep = self.ep0

        return ep

    def __set_ep_decay(self):
        """
        Creates a value for the exploration parameter based off of a decay heuristic.

        Returns
        --------
        ep: float
            The exploration parameter for the iteration

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Full decay is reached by 1/2 of the maximum number of BO iters
        """
        assert self.bo_iter is not None
        assert self.bo_iter_max - 1 >= self.bo_iter >= 0

        # Set ep_f to max value if it is too big
        # Initialize number of decay steps
        decay_steps = int(self.bo_iter_max / 2)
        # Apply heuristic on 1st iteration and all steps until end of decay steps
        if self.bo_iter < decay_steps or self.bo_iter == 0:
            ep = self.ep0 + (self.ep_f - self.ep0) * (self.bo_iter / self.bo_iter_max)
        else:
            ep = self.ep_f

        return ep

    def __set_ep_boyle(self):
        """
        Creates a value for the exploration parameter based on Boyle's Heuristic for GPO bounds

        Returns
        --------
        ep: float
            The exploration parameter for the iteration

        Notes
        -----
        Based on Heuristic from Boyle, P., Gaussian Processes for regression and Optimisation
        For these parameters, ep gets normalized between 0 and 2 given a neutral value of 1 as the starting point

        References
        ----------
        Boyle, P., Gaussian Processes for regression and Optimisation, Ph.D, Victoria University of Wellington, Wellington, New Zealand, 2007
        """
        # Set ep_curr as ep0 if it is not set
        if self.ep_curr is None:
            ep = self.ep0
        else:
            # Assert that improvement is not None
            assert self.improvement is not None
            # Apply a version of Boyle's heuristic
            # In original Boyle, you want to gradually expand or shrink your bounds
            # We take this concept for ep to increase exploration when improvement is FALSE and increase it when TRUE
            if self.improvement == True:
                # If we improved last time, Decrease exploration
                ep = self.ep_curr / self.ep_inc
            else:
                # If we did not, Increase Exploration
                ep = self.ep_curr * self.ep_inc

        # Ensure that ep stays within the bounds
        ep = self.__bound_ep(ep)

        return ep

    def __set_ep_jasrasaria(self):
        """
        Creates a value for the exploration parameter based off of Jasrasaria's heuristic

        Returns
        --------
        ep: float
            The exploration parameter for the iteration

        References
        ----------
        Heuristic from Jasrasaria, D., & Pyzer-Knapp, E. O. (2018). Dynamic Control of Explore/Exploit Trade-Off In Bayesian Optimization. http://arxiv.org/abs/1807.01279
        """
        assert self.best_error is not None
        assert self.mean_of_var is not None

        # Apply Jasrasaria's Heuristic
        if self.best_error > 0:
            ep = 1 + (self.mean_of_var / self.best_error**2)
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
    __init__(*): Constructor method
    """

    # Class variables and attributes
    def __init__(
        self,
        configuration,
        simulator_class,
        exp_data_class,
        list_gp_emulator_class,
        results_df,
        max_ei_details_df,
        why_term,
        heat_map_data_dict,
    ):
        """
        Parameters
        ----------
        configuration: dict
            Dictionary containing the configuration of the BO algorithm
        simulator_class: Simulator
            Class containing the Simulator class information
        exp_data_class: Data
            The experimental data for the workflow
        list_gp_emulator_class: list(GP_Emulator)
            Contains all gp_emulator information at each BO iter
        results_df: pd.DataFrame
            Dataframe including the values pertinent to BO for all BO runs
        max_ei_details_df: pd.DataFrame
            Dataframe including ei components of the best EI at each iter
        why_term: str
            String detailing the reason for algorithm termination
        heat_map_data_dict: dict
            Heat map data for each set of 2 parameters indexed by parameter names "param_1-param_2"
        """
        assert isinstance(configuration, dict) or configuration is None, "configuration must be a dictionary or None"
        assert isinstance(simulator_class, Simulator) or simulator_class is None, "simulator_class must be an instance of Simulator or None"
        assert isinstance(exp_data_class, Data) or exp_data_class is None, "exp_data_class must be an instance of Data or None"
        assert isinstance(list_gp_emulator_class, list) or list_gp_emulator_class is None, "list_gp_emulator_class must be a list or None"
        if list_gp_emulator_class is not None:
            assert all(isinstance(gp_emulator, (Type_1_GP_Emulator, Type_2_GP_Emulator)) for gp_emulator in 
                   list_gp_emulator_class), "entries of list list_gp_emulator_class must be Type_1_GP_Emulator or Type_2_GP_Emulator"
        assert isinstance(results_df, pd.DataFrame) or results_df is None, "results_df must be a pandas DataFrame or None"
        assert isinstance(max_ei_details_df, pd.DataFrame) or max_ei_details_df is None, "max_ei_details_df must be a pandas DataFrame or None"
        assert isinstance(why_term, (str, int)) or why_term is None, "why_term must be a string, int, or None"
        assert isinstance(heat_map_data_dict, dict) or heat_map_data_dict is None, "heat_map_data_dict must be a dictionary or None"
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
    __gen_start_pts_not_mc_sparse()
    __opt_with_scipy(opt_obj)
    __scipy_fxn(theta, opt_obj, best_error_metrics, beta)
    create_heat_map_param_data(n_points_set)
    __augment_train_data(theta_best_data)
    create_data_instance_from_theta(theta_array)
    __run_bo_iter(gp_model, iteration)
    __run_bo_to_term(gp_model)
    __run_bo_workflow()
    run_bo_restarts()
    """

    # Class variables and attributes

    def __init__(
        self,
        cs_params,
        method,
        simulator,
        exp_data,
        sim_data,
        sim_sse_data,
        val_data,
        val_sse_data,
        gp_emulator,
        ep_bias,
        gen_meth_theta,
    ):
        """
        Parameters
        ----------
        cs_params: CaseStudyParameters
            Class containing the values associated with CaseStudyParameters
        method: GPBO_Methods
            Class containing GPBO method information
        simulator: Simulator
            Class containing values of simulation parameters
        exp_data: Data
            Experimental data containing at least exp_data.theta_vals, exp_data.x_vals, and exp_data.y_vals
        sim_data: Data
            Simulated data containing at least sim_data.theta_vals, sim_data.x_vals, and sim_data.y_vals
        sim_sse_data: Data
            Simulated objective data containing at least sim_sse_data.theta_vals, sim_sse_data.x_vals, and sim_sse_data.y_vals
        val_data: Data or None
            Validation data containing at least val_data.theta_vals, val_data.x_vals, and val_data.y_vals
        val_sse_data: Data or None
            Validation data containing at least val_sse_data.theta_vals, val_sse_data.x_vals, and val_sse_data.y_vals
        gp_emulator: GP_Emulator
            Class containing gp_emulator data (set after training)
        ep_bias: Exploration_Bias class
            Class containing exploration parameter info
        gen_meth_theta: Gen_meth_enum or None
            The method by which simulation data is generated. For heat map making

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            cs_params, CaseStudyParameters
        ), "cs_params must be instance of CaseStudyParameters"
        assert isinstance(
            method, GPBO_Methods
        ), "method must be instance of GPBO_Methods"
        assert isinstance(
            simulator, Simulator
        ), "simulator must be instance of Simulator"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(sim_data, Data), "sim_data must be instance of Data"
        assert isinstance(sim_sse_data, Data), "sim_sse_data must be instance of Data"
        assert (
            isinstance(val_data, Data) or val_data is None
        ), "val_data must be instance of Data or None"
        assert (
            isinstance(val_sse_data, Data) or val_sse_data is None
        ), "val_sse_data must be instance of Data or None"
        assert (
            isinstance(gp_emulator, (Type_1_GP_Emulator, Type_2_GP_Emulator))
            or gp_emulator is None
        ), "gp_emulator must be instance of Type_1_GP_Emulator, Type_2_GP_Emulator, or None"
        assert isinstance(
            ep_bias, Exploration_Bias
        ), "ep_bias must be instance of Exploration_Bias"
        assert isinstance(
            gen_meth_theta, Gen_meth_enum
        ), "gen_meth_theta must be instance of Gen_meth_enum"

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
        self.bo_iter_term_frac = 0.3  # The fraction of iterations after which to terminate bo if no sse improvement is made
        self.sse_penalty = 1e7  # The penalty the __scipy_opt function gets for choosing nan theta values
        self.sg_mc_samples = 2000  # This can be changed at will
        # This object is used for optimization
        self.__min_obj_class = None
        # self.reset_rng()

    def __make_BO_results_temp(self, results_df, why_term, max_ei_details_df, list_gp_emulator_class):
        "Makes BO results from minimum data"

        # Set results for all compiled iterations for that run
        bo_results_res = BO_Results(
            None, None, self.exp_data, None, results_df, None, why_term, None
        )

        bo_results_GPs = BO_Results(
            None,
            None,
            None,
            list_gp_emulator_class,
            None,
            max_ei_details_df,
            None,
            None,
        )
        return bo_results_res, bo_results_GPs


    def __gen_emulator(self):
        """
        Sets GP Emulator class with training data and validation data based on the method class instance

        Returns
        --------
        gp_emulator: GP_Emulator
            Class for the GP emulator
        """
        # Determine Emulator Status, set gp_data data, and ininitalize correct GP_Emulator child class
        if self.method.emulator == False:
            all_gp_data = self.sim_sse_data
            all_val_data = self.val_sse_data
            k = np.maximum(self.exp_data.get_num_x_vals() - 1, 1)
            # If using objective sse use var of a chi^2 distribution (2k)
            if not self.method.obj.value == 2 and self.simulator.noise_std is not None:
                noise_scl_fact = np.sqrt(2 * k)
                noise_std = self.simulator.noise_std * noise_scl_fact
            # If using objective ln(sse) guess the noise std
            else:
                noise_std = None

            gp_emulator = Type_1_GP_Emulator(
                all_gp_data,
                all_val_data,
                None,
                None,
                None,
                self.cs_params.kernel,
                self.cs_params.lenscl,
                noise_std,
                self.cs_params.outputscl,
                self.cs_params.retrain_GP,
                self.cs_params.seed,
                self.cs_params.normalize,
                None,
                None,
                None,
                None,
            )
        else:
            all_gp_data = self.sim_data
            all_val_data = self.val_data
            noise_std = (
                self.simulator.noise_std
            )  # Yexp_std is exactly the noise_std of the GP Kernel
            gp_emulator = Type_2_GP_Emulator(
                all_gp_data,
                all_val_data,
                None,
                None,
                None,
                self.cs_params.kernel,
                self.cs_params.lenscl,
                noise_std,
                self.cs_params.outputscl,
                self.cs_params.retrain_GP,
                self.cs_params.seed,
                self.cs_params.normalize,
                None,
                None,
                None,
                None,
            )

        return gp_emulator

    def __get_best_error(self):
        """
        Helper function to calculate the best error and squared error calculations over x given the method.

        Returns
        -------
        be_data: Data
            Contains best_error as an instance of the data class
        be_metrics: tuple(float, np.ndarray, np.ndarray)
            The min_SSE, param at min_SSE, and squared residuals
        """

        if self.method.emulator == False:
            # Type 1 best error is inferred from training data
            best_error, be_theta, train_idx = self.gp_emulator.calc_best_error()
            best_errors_x = None
            be_data = self.create_data_instance_from_theta(
                be_theta.flatten(), get_y=False
            )
            be_data.y_vals = np.atleast_1d(
                self.gp_emulator.train_data.y_vals[train_idx]
            )
        else:
            # Type 2 best error must be calculated given the experimental data
            best_error, be_theta, best_errors_x, train_idx = (
                self.gp_emulator.calc_best_error(self.method, self.exp_data)
            )
            be_data = self.create_data_instance_from_theta(
                be_theta.flatten(), get_y=False
            )
            be_data.y_vals = self.gp_emulator.train_data.y_vals[
                train_idx[0] : train_idx[1]
            ]

        be_metrics = best_error, be_theta, best_errors_x

        return be_data, be_metrics

    def __make_starting_opt_pts(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.__get_best_error()

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
        """
        # Note: Could make this generate 2 sets of starting points based on whether you want to optimize sse or ei
        # For sparse grid and mc methods
        if self.method.sparse_grid == True or self.method.mc == True:
            starting_pts = self.__gen_start_pts_mc_sparse(best_error_metrics, rng_seed)
        else:
            starting_pts = self.__gen_start_pts_not_mc_sparse(rng_seed)

        return starting_pts

    def __gen_start_pts_mc_sparse(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy if using sparse grid or Monte Carlo methods

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
        """
        # Generate n LHS Theta vals
        num_mc_theta = 500
        theta_vals = self.simulator.gen_theta_vals(num_mc_theta, rng_seed)

        # Add repeated theta_vals and experimental x values
        rep_theta_vals = np.repeat(theta_vals, len(self.exp_data.x_vals), axis=0)
        rep_x_vals = np.vstack([self.exp_data.x_vals] * num_mc_theta)

        # Create instance of Data Class
        sp_data = Data(
            rep_theta_vals,
            rep_x_vals,
            None,
            None,
            None,
            None,
            None,
            None,
            self.simulator.bounds_theta_reg,
            self.simulator.bounds_x,
            self.cs_params.sep_fact,
        )

        # Evaluate GP mean and Var (This is the slowest step)
        feat_sp_data = self.gp_emulator.featurize_data(sp_data)
        sp_data.gp_mean, sp_data.gp_var = self.gp_emulator.eval_gp_mean_var_misc(
            sp_data, feat_sp_data
        )

        # Evaluate GP SSE and SSE_Var (This is the 2nd slowest step)
        sp_data_sse_mean, sp_data_sse_var = self.gp_emulator.eval_gp_sse_var_misc(
            sp_data, self.method, self.exp_data
        )

        # Note - Use Sparse grid EI for approximations
        # Evaluate EI using Sparse Grid or EI (This is relatively quick)
        method_3 = GPBO_Methods(Method_name_enum(3))
        sp_data_ei, iter_max_ei_terms = self.gp_emulator.eval_ei_misc(
            sp_data, self.exp_data, self.ep_bias, best_error_metrics, method_3
        )

        ##Sort by min(-ei)
        # Create a list of tuples containing indices and values
        indexed_values = list(enumerate(-1 * sp_data_ei))  # argmin(-ei) = argmax(ei)

        # Sort the list of tuples based on values
        sorted_values = sorted(indexed_values, key=lambda x: x[1])

        # Extract the indices from the sorted list
        min_indices = [index for index, _ in sorted_values]
        # Sets the points in order based on the indices
        all_pts = theta_vals[min_indices]

        # Choose top retrain_GP points as starting points
        starting_pts = all_pts[: self.cs_params.reoptimize_obj + 1]

        return starting_pts

    def __gen_start_pts_not_mc_sparse(self, rng_seed):
        """
        Makes starting point for optimization with scipy if not using sparse grid or Monte Carlo methods

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
        """
        # Create starting points equal to number of retrain_GP
        starting_pts = self.simulator.gen_theta_vals(
            self.cs_params.reoptimize_obj + 1, rng_seed
        )

        return starting_pts

    def __opt_with_scipy(self, opt_obj, get_y = False, w_noise = False):
        """
        Optimizes a function with scipy.optimize

        Parameters
        ----------
        opt_obj: str
            Which objective to calculate. neg_ei, E[SSE], or SSE
        get_y: bool, default False
            Whether to return the y values of the optimized theta
        w_noise: bool, default False
            Whether to return the y values with noise

        Returns
        --------
        best_val: float
            The optimized value of the function
        best_theta: np.ndarray
            The parameter set corresponding to best_val

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        self.__min_obj_class = None

        assert isinstance(opt_obj, str), "opt_obj must be string!"
        assert opt_obj in [
            "neg_ei",
            "E_sse",
            "sse",
        ], "opt_obj must be 'neg_ei', or 'sse'!"

        # Note use > because index 0 counts as 1 reoptimization
        if self.cs_params.reoptimize_obj > 50:
            warnings.warn("The objective will be reoptimized more than 50 times!")

        # Calc best error
        be_data, best_error_metrics = self.__get_best_error()

        # Find bounds and arguments for function
        bnds = (
            self.simulator.bounds_theta_reg.T
        )  # Transpose bounds to work with scipy.optimize
        # Need to account for normalization here (make bounds array of [0,1]^dim_theta)

        ## Loop over each validation point/ a certain number of validation point thetas
        for i in range(self.cs_params.reoptimize_obj + 1):
            # Choose a random index of theta to start with
            theta_guess = self.opt_start_pts[i].flatten()

            # Initialize L-BFGS-B as default optimization method
            obj_opt_method = "L-BFGS-B"

            try:
                # Call scipy method to optimize EI given theta
                # Using L-BFGS-B instead of BFGS because it allowd for bounds
                best_result = optimize.minimize(
                    self.__scipy_fxn,
                    theta_guess,
                    bounds=bnds,
                    method=obj_opt_method,
                    args=(opt_obj, best_error_metrics),
                )
            except ValueError:
                # If the intialized theta causes scipy.optimize to choose nan values, skip it
                pass

        best_val = self.__min_obj_class.acq
        best_class = self.__min_obj_class

        best_class_simple = self.create_data_instance_from_theta(
            self.__min_obj_class.theta_vals[0], get_y=get_y, w_noise=w_noise
        )
        if get_y:
            best_class.y_vals = best_class_simple.y_vals

        return best_val, best_class

    def __scipy_fxn(self, theta, opt_obj, best_error_metrics):
        """
        Calculates either -ei, sse objective, or E[SSE] at a candidate parameter set value

        Parameters
        -----------
        theta: np.ndarray
            Array of theta values to optimize
        opt_obj: str
            Which objective to calculate. 'neg_ei', 'E_sse', or 'sse'
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.__get_best_error()

        Returns
        --------
        obj: float, the value of the specified objective function for the given candidate parameter set

        Notes
        -----
        If there are nan values in theta, the objective function is set to 1 for neg_ei and self.sse_penalty for sse and E_sse

        """
        rng = self.rng_set
        # Set seed
        # Check if there are nan values in theta
        if np.isnan(theta).any():
            # If there are nan values, set neg ei to 1 (ei = -1)
            if opt_obj == "neg_ei":
                obj = 1
            # Set sse and lcb to self.sse_penalty
            else:
                obj = self.sse_penalty

        # If not, continue the algorithm normally
        else:
            candidate = Data(
                None,
                self.exp_data.x_vals,
                None,
                None,
                None,
                None,
                None,
                None,
                self.simulator.bounds_theta_reg,
                self.simulator.bounds_x,
                self.cs_params.sep_fact,
            )

            # Create feature data for candidate point
            if self.method.emulator == False:
                candidate_theta_vals = theta.reshape(1, -1)
            else:
                candidate_theta_vals = np.repeat(
                    theta.reshape(1, -1), self.exp_data.get_num_x_vals(), axis=0
                )

            candidate.theta_vals = candidate_theta_vals
            self.gp_emulator.cand_data = candidate

            # Set candidate point feature data
            self.gp_emulator.feature_cand_data = self.gp_emulator.featurize_data(
                self.gp_emulator.cand_data
            )

            # Evaluate GP mean/ stdev at theta
            cand_mean, cand_var = self.gp_emulator.eval_gp_mean_var_cand()

            # Evaluate SSE & SSE stdev at theta
            if self.method.emulator == False:
                # For Type 1 GP, the sse and sse_var are directly inferred from the gp_mean and gp_var
                cand_sse_mean, cand_sse_var = self.gp_emulator.eval_gp_sse_var_cand()
            else:
                # For Type 2 GP, the sse and sse_var are calculated from the gp_mean, gp_var, and experimental data
                cand_sse_mean, cand_sse_var = self.gp_emulator.eval_gp_sse_var_cand(
                    self.method, self.exp_data
                )

            # Calculate objective fxn
            if opt_obj == "sse":
                # Objective to minimize is log(sse) if using 1B, and sse for all other methods
                obj = cand_sse_mean
            elif opt_obj == "E_sse":
                # Objective to minimize is (E)[sse] for method ESSE
                obj = cand_sse_mean + np.sum(cand_sse_var)
            else:
                # Otherwise objective is ei
                if self.method.emulator == False:
                    ei_output = self.gp_emulator.eval_ei_cand(
                        self.exp_data, self.ep_bias, best_error_metrics
                    )
                else:
                    ei_output = self.gp_emulator.eval_ei_cand(
                        self.exp_data,
                        self.ep_bias,
                        best_error_metrics,
                        self.method,
                        self.sg_mc_samples,
                    )
                obj = -1 * ei_output[0]

            set_acq_val = True

            # Save candidate class if there is no current value
            if self.__min_obj_class == None:
                self.__min_obj_class = self.gp_emulator.cand_data
            # The sse/lcb objective is smaller than what we have so far
            elif self.__min_obj_class.acq > obj and opt_obj != "neg_ei":
                self.__min_obj_class = self.gp_emulator.cand_data
            # The ei objective is larger than what we have so far
            elif self.__min_obj_class.acq * -1 > obj and opt_obj == "neg_ei":
                self.__min_obj_class = self.gp_emulator.cand_data
            # For SSE, if the objective is the same, randomly choose between the two (since sse is an objective fxn)
            elif (
                np.isclose(self.__min_obj_class.acq, obj, rtol=1e-7)
                and opt_obj == "sse"
            ):
                # random_number = rng.randint(0, 1)
                random_number = rng.integers(0,1)
                if random_number > 0:
                    self.__min_obj_class = self.gp_emulator.cand_data
                else:
                    set_acq_val = False
            # For EI/E_sse (acquisition fxns) switch to the value farthest from any training data
            elif np.isclose(self.__min_obj_class.acq, obj, rtol=1e-7):
                # Get the distance between the candidate and the current min_obj_class value and the training data
                dist_old = (
                    distance.cdist(
                        self.gp_emulator.train_data.theta_vals,
                        self.__min_obj_class.theta_vals[0, :].reshape(1, -1),
                        metric="euclidean",
                    )
                    .ravel()
                    .max()
                )
                dist_new = (
                    distance.cdist(
                        self.gp_emulator.train_data.theta_vals,
                        self.gp_emulator.cand_data.theta_vals[0, :].reshape(1, -1),
                        metric="euclidean",
                    )
                    .ravel()
                    .max()
                )
                # If the distance of the new point is larger or equal to the old point, keep the new point
                if dist_new >= dist_old:
                    self.__min_obj_class = self.gp_emulator.cand_data
                else:
                    set_acq_val = False
            else:
                set_acq_val = False

            if set_acq_val and opt_obj != "neg_ei":
                self.__min_obj_class.acq = obj

        return obj

    def create_heat_map_param_data(self, n_points_set=None):
        """
        Creates parameter sets that can be used to generate heat maps of data at any given iteration

        Parameters
        -----------
        n_points_set: int or None, default None
            The number of points to use per axis for creating heat maps. If None, the number of unique simulation points is used

        Returns
        --------
        heat_map_data_dict: dict
            Heat map data for each set of 2 parameters indexed by parameter name tuple ("param_1,param_2")
        """
        assert isinstance(
            self.gp_emulator, (Type_1_GP_Emulator, Type_2_GP_Emulator)
        ), "self.gp_emulator must be instance of Type_1_GP_Emulator or Type_2_GP_Emulator"
        assert isinstance(
            self.gp_emulator.gp_sim_data, Data
        ), "self.gp_emulator.gp_sim_data must be an instance of Data!"
        assert isinstance(
            self.gen_meth_theta, Gen_meth_enum
        ), "self.gen_meth_theta must be instance of Gen_meth_enum"
        assert isinstance(
            self.exp_data.x_vals, (np.ndarray)
        ), "self.exp_data.x_vals must be np.ndarray"
        assert (
            isinstance(n_points_set, int) or n_points_set is None
        ), "n_points_set must be None or int"

        # Create list of heat map theta data
        heat_map_data_dict = {}

        # Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(
            0, self.simulator.dim_theta - 1, self.simulator.dim_theta
        )
        # Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype=int)

        # Set x_vals
        norm_x_vals = self.exp_data.x_vals
        num_x = self.exp_data.get_num_x_vals()

        # If no number of points is set, use the length of the unique simulation thetas
        if n_points_set == None:
            # Use number of training theta for number of theta points
            n_thetas_points = len(self.gp_emulator.gp_sim_data.get_unique_theta())
            # Initialze meshgrid-like set of theta values at their true values
            # If points were generated with an LHS, the number of points per parameter is n_thetas_points for the meshgrid
            if self.gen_meth_theta.value == 1:
                n_points = n_thetas_points
            else:
                # For a meshgrid, the number of theta values/ parameter is n_thetas_points for the meshgrid ^(1/theta_dim)
                n_points = int((n_thetas_points) ** (1 / self.simulator.dim_theta))
        else:
            n_points = n_points_set

        # Ensure we will never generate more than 5000 pts per heat map
        # if self.method.emulator == True:
        if num_x * n_points**2 >= 5000:
            n_points = int(np.sqrt(5000 / (num_x)))

        # Meshgrid set always defined by n_points**2
        # Set thetas for meshgrid. Never use more than 10000 points
        theta_set = np.tile(np.array(self.simulator.theta_true), (n_points**2, 1))

        # Infer how many times to repeat theta and x values given that heat maps are meshgrid form by definition
        # The meshgrid of parameter values created below is symmetric, therefore, x is repeated by n_points**2 for a 2D meshgrid
        repeat_x = n_points**2  # Square because only 2 values at a time change
        x_vals = np.vstack([norm_x_vals] * repeat_x)
        repeat_theta = self.exp_data.get_num_x_vals()

        # Loop over all possible theta combinations of 2
        for i in range(len(mesh_combos)):
            # Create a copy of the true values to change the mehsgrid valus on
            theta_set_copy = np.copy(theta_set)
            # Set the indeces of theta_set for evaluation as each row of mesh_combos
            idcs = mesh_combos[i]
            # define name of parameter set as tuple ("param_1,param_2")
            data_set_name = (
                self.simulator.theta_true_names[idcs[0]],
                self.simulator.theta_true_names[idcs[1]],
            )

            # Create a meshgrid of values of the 2 selected values of theta and reshape to the correct shape
            # Assume that theta1 and theta2 have equal number of points on the meshgrid
            theta1 = np.linspace(
                self.simulator.bounds_theta_reg[0][idcs[0]],
                self.simulator.bounds_theta_reg[1][idcs[0]],
                n_points,
            )
            theta2 = np.linspace(
                self.simulator.bounds_theta_reg[0][idcs[1]],
                self.simulator.bounds_theta_reg[1][idcs[1]],
                n_points,
            )
            theta12_mesh = np.array(np.meshgrid(theta1, theta2))
            theta12_vals = np.array(theta12_mesh).T.reshape(-1, 2)

            # Set initial values for evaluation (true values) to meshgrid values
            theta_set_copy[:, idcs] = theta12_vals

            # Put values into instance of data class
            # Create data set based on emulator status
            if self.method.emulator == True:
                # Repeat the theta vals for Type 2 methods to ensure that theta and x values are in the correct form for evaluation with gp_emulator.eval_gp_mean_heat_map()
                theta_vals = np.repeat(theta_set_copy, repeat_theta, axis=0)
                data_set = Data(
                    theta_vals,
                    x_vals,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.simulator.bounds_theta_reg,
                    self.simulator.bounds_x,
                    self.cs_params.sep_fact,
                )
            else:
                data_set = Data(
                    theta_set_copy,
                    norm_x_vals,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.simulator.bounds_theta_reg,
                    self.simulator.bounds_x,
                    self.cs_params.sep_fact,
                )

            # Append data set to dictionary with name
            heat_map_data_dict[data_set_name] = data_set

        return heat_map_data_dict

    def __augment_train_data(self, theta_best_data):
        """
        Augments training data given a new data point

        Parameters
        ----------
        theta_best_data: Data
            The parameter set data associated with the optimal acquisition function value
        """
        # Augment training theta, x, and y/sse data
        self.gp_emulator.add_next_theta_to_train_data(theta_best_data)

    def create_data_instance_from_theta(self, theta_array, get_y=True, w_noise = False):
        """
        Creates instance of Data from an np.ndarray parameter set

        Parameters
        ----------
        theta_array: np.ndarray
            Array of parameter values to turn into an instance of Data
        get_y: bool, default True
            Whether to calculate y values for theta_array
        w_noise: bool, default False
            Whether to add noise to the y values

        Returns
        --------
        theta_arr_data: Data
            Data class instance for the theta_array

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        rng = self.rng_set

        assert isinstance(theta_array, np.ndarray), "theta_array must be np.ndarray"
        assert len(theta_array.shape) == 1, "theta_array must be 1D"
        assert isinstance(
            self.exp_data.x_vals, (np.ndarray)
        ), "self.exp_data.x_vals must be np.ndarray"

        # Repeat the theta best array once for each x value
        # Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.gen_y_data
        theta_arr_repeated = np.repeat(
            theta_array.reshape(1, -1), self.exp_data.get_num_x_vals(), axis=0
        )
        # Add instance of Data class to theta_best
        theta_arr_data = Data(
            theta_arr_repeated,
            self.exp_data.x_vals,
            None,
            None,
            None,
            None,
            None,
            None,
            self.simulator.bounds_theta_reg,
            self.simulator.bounds_x,
            self.cs_params.sep_fact,
        )
        if get_y:
            if w_noise:
                # Calculate y values and sse for theta_best with noise
                theta_arr_data.y_vals = self.simulator.gen_y_data(
                    theta_arr_data, self.simulator.noise_mean, self.simulator.noise_std, rng
                )
            else:
                # Calculate y values and sse for theta_best without noise
                theta_arr_data.y_vals = self.simulator.gen_y_data(
                    theta_arr_data, self.simulator.noise_mean, 0, rng
                )

        # Set the best data to be in sse form if using a type 1 GP
        if self.method.emulator == False:
            theta_arr_data = self.simulator.sim_data_to_sse_sim_data(
                self.method,
                theta_arr_data,
                self.exp_data,
                self.cs_params.sep_fact,
                not get_y,
            )

        return theta_arr_data

    def __run_bo_iter(self, iteration):
        """
        Runs a single GPBO iteration

        Parameters
        ----------
        iteration: int, The iteration of BO in progress

        Returns
        --------
        iter_df: pd.DataFrame
            Dataframe containing the results from the GPBO Workflow for iteration
        iter_max_ei_terms: pd.DataFrame or None
            Contains ei calculation terms for max ei parameter set if self.cs_params.save_data
        gp_emulator_curr: GP_Emulator
            The class used for this iteration of the GPBO workflow
        """
        # Start timer
        # Initialize iter_max_ei df to None
        iter_max_ei_terms = None
        
        #Initialize the iterations seed with start_seed as a backup
        iter_seed = self.simulator.start_seed

        #Generate a random number for the seed to generate initial LHS samples with that is not the same as the sim or val seeds 
        if self.cs_params.seed is not None:
            for i in range(10):
                seed_init = self.rng_set.integers(1, 1e8)
                if seed_init not in [self.simulator.sim_seed, self.simulator.val_seed]:
                    iter_seed = seed_init
                    break
        else:
            iter_seed = None
        
        time_start = time.time()

        # Train GP model (this step updates the model to a trained model)
        self.gp_emulator.train_gp()

        # Calcuate best error
        best_err_data, best_error_metrics = self.__get_best_error()

        # Add not log best error to ep_bias
        if iteration == 0 or self.ep_bias.ep_enum.value == 4:
            # Since best error is squared when used in Jasrasaria calculations, the value will always be >=0
            self.ep_bias.best_error = best_error_metrics[0]

        # Calculate mean of var for validation set if using Jasrasaria heuristic
        if self.ep_bias.ep_enum.value == 4:
            # Calculate average gp mean and variance of the validation set
            val_gp_mean, val_gp_var = self.gp_emulator.eval_gp_mean_var_val()
            # For emulator methods, the mean of the variance should come from the sse variance
            if self.method.emulator == True:
                # Redefine gp_mean and gp_var to be the mean and variane of the sse
                val_gp_mean, val_gp_var = self.gp_emulator.eval_gp_sse_var_val(
                    self.method, self.exp_data
                )

            # Check for ln(sse) values
            # For 1B, propogate errors associated with an unlogged sse value
            val_gp_var = val_gp_var * np.exp(val_gp_mean) ** 2

            # Set mean of sse variance
            mean_of_var = np.average(val_gp_var)
            self.ep_bias.mean_of_var = mean_of_var

        # Set initial exploration bias and bo_iter
        if self.ep_bias.ep_enum.value == 2:
            self.ep_bias.bo_iter = iteration

        # Calculate new ep. Note. It is extemely important to do this AFTER setting the ep_max
        self.ep_bias.set_ep()

        # Set Optimization starting points for this iteration
        self.opt_start_pts = self.__make_starting_opt_pts(best_error_metrics, iter_seed)

        # Call optimize E[SSE] or log(E[SSE]) objective function
        # Note if we didn't want actual sse values, we would have to set get_y_sse = False
        min_sse, min_theta_data = self.__opt_with_scipy("sse", get_y = self.cs_params.get_y_sse, w_noise = self.cs_params.w_noise)

        # Call optimize EI acquistion fxn (If not using E[SSE])
        if self.method.method_name.value != 7:
            opt_acq, acq_theta_data = self.__opt_with_scipy("neg_ei", get_y = True, w_noise = self.cs_params.w_noise)
            if self.method.emulator == True:
                ei_args = (
                    acq_theta_data,
                    self.exp_data,
                    self.ep_bias,
                    best_error_metrics,
                    self.method,
                    self.sg_mc_samples,
                )
            else:
                ei_args = (
                    acq_theta_data,
                    self.exp_data,
                    self.ep_bias,
                    best_error_metrics,
                )
        else:
            opt_acq, acq_theta_data = self.__opt_with_scipy("E_sse", get_y = True, w_noise = self.cs_params.w_noise)

        # If type 2, turn it into sse_data
        # Set the best data to be in sse form if using a type 2 GP and find the min sse
        if self.method.emulator == True:
            # Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = self.simulator.sim_data_to_sse_sim_data(
                self.method,
                min_theta_data,
                self.exp_data,
                self.cs_params.sep_fact,
                False,
            )
            acq_sse_theta_data = self.simulator.sim_data_to_sse_sim_data(
                self.method,
                min_theta_data,
                self.exp_data,
                self.cs_params.sep_fact,
                False,
            )

        # Otherwise the sse data is the original (scaled) data
        else:
            # Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = min_theta_data
            acq_sse_theta_data = acq_theta_data

        # Evaluate max EI terms at theta
        if self.cs_params.save_data and not self.method.method_name.value == 7:
            ei_max, iter_max_ei_terms = self.gp_emulator.eval_ei_misc(*ei_args)

        # Turn min_sse_sim value into a float (this makes analyzing data from csvs and dataframes easier)
        min_sse_gp = float(min_sse)
        opt_acq_sim = float(acq_sse_theta_data.y_vals)

        # calculate improvement if using Boyle's method to update the exploration bias
        # Improvement is true if the min sim sse found is lower than (not log) best error, otherwise it's false
        if min_sse_gp < best_error_metrics[0]:
            improvement = True
        else:
            improvement = False
        if self.ep_bias.ep_enum.value == 3:
            # Set ep improvement
            self.ep_bias.improvement = improvement

        # Create a copy of the GP Emulator Class for this iteration
        gp_emulator_curr = copy.deepcopy(self.gp_emulator)

        # Call __augment_train_data to append training data
        self.__augment_train_data(acq_theta_data)


        # Calc time/ iter
        time_end = time.time()
        time_per_iter = time_end - time_start

        # Create Results Pandas DataFrame for 1 iter
        column_names = [
            "Best Error",
            "Exploration Bias",
            "Theta Opt Acq",
            "Opt Acq",
            "Acq Obj Act",
            "MSE Acq Act",
            "Theta Min Obj",
            "Min Obj GP",
            "Min Obj Act",
            "MSE Obj GP",
            "MSE Obj Act",
            "Time/Iter",
        ]
        num_exp_x = self.exp_data.get_num_x_vals()
        # Return SSE and not log(SSE) for 'Min Obj', 'Min Obj Act', 'Min Obj GP' when calculating MSE
        MSE_acq_obj_act = (
            np.exp(opt_acq_sim) / num_exp_x
            if self.method.obj.value == 2
            else opt_acq_sim / num_exp_x
        )
        if self.cs_params.get_y_sse:
            min_sse_sim = float(min_sse_theta_data.y_vals)
            MSE_obj_act = (
                np.exp(min_sse_sim) / num_exp_x
                if self.method.obj.value == 2
                else min_sse_sim / num_exp_x
            )
        else:
            min_sse_sim = None
            MSE_obj_act = None

        MSE_obj_gp = (
            np.exp(min_sse_gp) / num_exp_x
            if self.method.obj.value == 2
            else min_sse_gp / num_exp_x
        )
        bo_iter_results = [
            best_error_metrics[0],
            float(self.ep_bias.ep_curr),
            acq_theta_data.theta_vals[0],
            float(opt_acq),
            opt_acq_sim,
            MSE_acq_obj_act,
            min_sse_theta_data.theta_vals[0],
            min_sse_gp,
            min_sse_sim,
            MSE_obj_gp,
            MSE_obj_act,
            time_per_iter,
        ]
        iter_df = pd.DataFrame(columns=column_names)
        # Add the new row to the DataFrame
        iter_df.loc[0] = bo_iter_results

        return iter_df, iter_max_ei_terms, gp_emulator_curr

    def __run_bo_to_term(self, run_num, job = None):
        """
        Runs GPBO to termination

        Params:
        -------
        gp_model: gpflow.models.GPR, GP emulator for workflow

        Returns
        --------
        iter_df: pd.DataFrame
            Dataframe containing the results from the GPBO Workflow for all iterations
        max_ei_details_df: pd.DataFrame
            Contains ei data for max ei parameter sets for each bo iter if self.cs_params.save_data
        list_gp_emulator_class: list(GP_Emulator)
            The classes used for all iterations of the GPBO workflow
        why_term: str
            String containing reasons for bo algorithm termination

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert (
            0 < self.bo_iter_term_frac <= 1
        ), "self.bo_iter_term_frac must be between 0 and 1"
        # Initialize pandas dataframes
        column_names = [
            "Best Error",
            "Exploration Bias",
            "Theta Opt Acq",
            "Opt Acq",
            "Acq Obj Act",
            "MSE Acq Act",
            "Theta Min Obj",
            "Min Obj GP",
            "Min Obj Act",
            "MSE Obj GP",
            "MSE Obj Act",
            "Time/Iter",
        ]

        results_df = pd.DataFrame(columns=column_names)
        max_ei_details_df = pd.DataFrame()
        list_gp_emulator_class = []
        # Initialize count
        obj_counter = 0
        self.ep_bias.ep_curr = None

        cond1 = len(self.gpbo_res_GP) > 0
        cond2 = len(self.gpbo_res_GP) == run_num + 1
        cond3 = job is not None

        #Check for files in job for gpbo_res_simple and gpbo_res_GP
        if cond1 and cond2 and cond3:
            results_df = self.gpbo_res_simple[run_num].results_df
            self.ep_bias.ep_curr = self.gpbo_res_simple[run_num].results_df["Exploration Bias"].iloc[-1]
            #The obj_counter is set as why_term until termination happens
            obj_counter = self.gpbo_res_simple[run_num].why_term
            list_gp_emulator_class = self.gpbo_res_GP[run_num].list_gp_emulator_class

        #Start at the next iteration after data ends. If no data, start at 0
        iter_start = len(list_gp_emulator_class)

        # Initilize terminate flags
        acq_flag = False
        obj_flag = False
        terminate = False

        # Set why_term strings
        why_terms = ["acq", "obj", "max_budget"]

        # Do Bo iters while stopping criteria is not met
        while terminate == False:
            # Loop over number of max bo iters
            for i in range(iter_start, self.cs_params.bo_iter_tot, 1):
                # Output results of 1 bo iter and the emulator used to get the results
                iter_df, iter_max_ei_terms, gp_emulator_class = self.__run_bo_iter(
                    i
                ) 
                # Add results to dataframe
                results_df = pd.concat(
                    [results_df.astype(iter_df.dtypes), iter_df], ignore_index=True
                )
                if iter_max_ei_terms is not None:
                    max_ei_details_df = pd.concat(
                        [max_ei_details_df, iter_max_ei_terms]
                    )
                # At the first iteration
                if i == 0:
                    # improvement is defined as infinity on 1st iteration (something is always better than nothing)
                    improvement = np.inf
                elif results_df["Min Obj GP"].iloc[i] < float(
                    results_df["Min Obj GP"][:-1].min()
                ):
                    # And the improvement is defined as the difference between the last Min Obj Cum. and current Obj Min (unscaled)
                    if self.method.obj.value == 1:
                        improvement = (
                            results_df["Min Obj GP"][:-1].min()
                            - results_df["Min Obj GP"].iloc[i]
                        )
                    else:
                        improvement = np.exp(
                            results_df["Min Obj GP"][:-1].min()
                        ) - np.exp(results_df["Min Obj GP"].iloc[i])
                # Otherwise
                else:
                    # And the improvement is defined as 0, since it must be non-negative
                    improvement = 0

                # Add gp emulator data from that iteration to list (before stopping criteria)
                list_gp_emulator_class.append(gp_emulator_class)
                # Call stopping criteria after 1st iteration and update improvement counter
                # If the improvement is negligible, add to counter
                if improvement < self.cs_params.obj_tol:
                    obj_counter += 1
                # Otherwise reset the counter
                else:
                    obj_counter = 0

                # set flag if opt acq. func val is less than the tolerance 3 times in a row
                if (
                    all(results_df["Opt Acq"].tail(3) < self.cs_params.acq_tol)
                    and i >= 4
                ):
                    acq_flag = True
                # set flag if small sse progress over 1/3 of total iteration budget
                if (
                    obj_counter
                    >= int(self.cs_params.bo_iter_tot * self.bo_iter_term_frac)
                    and i >= 4
                ):
                    obj_flag = True

                flags = [acq_flag, obj_flag]

                # Terminate if you meet 2 stopping criteria, hit the budget, or obj has not improved after 1/2 of iterations
                if i == self.cs_params.bo_iter_tot - 1:
                    terminate = True
                    why_term = why_terms[-1]
                    break
                elif flags.count(True) >= 2:
                    terminate = True
                    # Pull indecies of list that are true
                    term_flags = [
                        why_terms[index] for index, value in enumerate(flags) if value
                    ]
                    why_term = "-".join(term_flags)
                    break
                elif (
                    obj_counter >= int(self.cs_params.bo_iter_tot * 0.5)
                    and self.cs_params.bo_iter_tot >= 5
                ):
                    terminate = True
                    why_term = why_terms[1]
                    break
                # Continue if no stopping criteria are met
                else:
                    terminate = False
                
                #Save results
                #make a new list of emulator classes to save which includes a copy of the original list + newest GP object
                list_emulator_class_temp = copy.deepcopy(list_gp_emulator_class.copy())
                list_emulator_class_temp[-1] = copy.deepcopy(self.gp_emulator)
                #Make temporary BO results for this iter
                bo_results_res, bo_results_GPs = self.__make_BO_results_temp(results_df, obj_counter, max_ei_details_df, list_emulator_class_temp)
                # Add simulator class and save the rng seeds that are being used
                bo_results_res.simulator_class = copy.deepcopy(self.simulator)
                bo_results_GPs.driver_rng = copy.deepcopy(self.rng_set)
                bo_results_res.sim_rng = copy.deepcopy(self.simulator.rng_set)
                
                #Save results at each iteration so that if the job takes a while it can be continued
                if len(self.gpbo_res_simple) == len(self.gpbo_res_GP) != run_num + 1:
                    self.gpbo_res_simple.append(bo_results_res)
                    self.gpbo_res_GP.append(bo_results_GPs)
                else:
                    self.gpbo_res_simple[run_num] = bo_results_res
                    self.gpbo_res_GP[run_num] = bo_results_GPs

                #Save results
                if job is not None:
                    self.save_results_run(job)
                
        # Reset the index of the pandas df
        results_df = results_df.reset_index()

        # Fill Cumulative value columns based on results
        # Initialize cum columns as the same as the original columns
        results_df.rename(columns={"index": "BO Iter"}, inplace=True)
        results_df["BO Iter"] += 1
        results_df["BO Method"] = self.method.report_name
        results_df["Max Evals"] = len(results_df)
        results_df["Theta Acq Act Cum"] = results_df["Theta Opt Acq"]
        results_df["Theta Obj GP Cum"] = results_df["Theta Min Obj"]
        results_df["Theta Obj Act Cum"] = results_df["Theta Min Obj"]
        results_df["Termination"] = why_term
        results_df["Total Run Time"] = float(results_df["Time/Iter"].sum())

        results_df["Min Obj GP Cum"] = np.minimum.accumulate(results_df["Min Obj GP"])
        if self.cs_params.get_y_sse == True:
            results_df["Min Obj Act Cum"] = np.minimum.accumulate(results_df["Min Obj Act"])
        else:
            results_df["Min Obj Act Cum"] = None
        results_df["Acq Obj Act Cum"] = np.minimum.accumulate(results_df["Acq Obj Act"])

        # Add cumulative values to the dataframe
        for i in range(len(results_df)):
            if i > 0:
                if (
                    results_df["Acq Obj Act Cum"].iloc[i]
                    >= results_df["Acq Obj Act Cum"].iloc[i - 1]
                ):
                    results_df.at[i, "Theta Acq Act Cum"] = (
                        results_df["Theta Acq Act Cum"].iloc[i - 1].copy()
                    )
                #If we are tracking actual values, update as normal, otherwise follow the same trend os the GP SSE
                if (self.cs_params.get_y_sse == True and
                    results_df["Min Obj Act Cum"].iloc[i]
                    >= results_df["Min Obj Act Cum"].iloc[i - 1] 
                ) or (self.cs_params.get_y_sse == False and results_df["Min Obj GP Cum"].iloc[i]
                    >= results_df["Min Obj GP Cum"].iloc[i - 1]):
                    results_df.at[i, "Theta Obj Act Cum"] = (
                        results_df["Theta Obj Act Cum"].iloc[i - 1].copy()
                    )
                if (
                    results_df["Min Obj GP Cum"].iloc[i]
                    >= results_df["Min Obj GP Cum"].iloc[i - 1]
                ):
                    results_df.at[i, "Theta Obj GP Cum"] = (
                        results_df["Theta Obj GP Cum"].iloc[i - 1].copy()
                    )

        # Create df for ei and add those results here
        if iter_max_ei_terms is not None:
            max_ei_details_df.columns = iter_max_ei_terms.columns.tolist()
            max_ei_details_df = max_ei_details_df.reset_index(drop=True)


        return results_df, max_ei_details_df, list_gp_emulator_class, why_term

    def __run_bo_workflow(self, run_num, job = None):
        """
        Runs a GPBO method through all bo iterations and reports the data for that run of the method

        Returns
        --------
        bo_results_res: BO_Results
            Includes table of results, exp_Data and why term for the GPBO workflow
        bo_results_GPs: BO_Results
            Includes the GP emulator classes used and max ei details for each iteration of the BO workflow

        Notes
        ------
        Two instances of BO_Results are used since opening the GP files is often tedious and we may not need to open them to analyze the results
        """
        
        #If a results object for this run exists, load it
        cond1 = len(self.gpbo_res_GP) > 0
        cond2 = len(self.gpbo_res_GP) == run_num + 1
        cond3 = job is not None

        #If results exist for this run and is being saved, use the emulator class from the last iteration
        if cond1 and cond2 and cond3:
            self.gp_emulator = self.gpbo_res_GP[run_num].list_gp_emulator_class[-1]
            self.rng_set = self.gpbo_res_GP[run_num].driver_rng
            self.simulator.rng_set = self.gpbo_res_simple[run_num].sim_rng
        #If results do not exist for this run, initialize the emulator class
        else:
            #Reset driver rng at each run to update seed for driver class
            self.reset_rng()
            # Initialize gp_emualtor class
            gp_emulator = self.__gen_emulator()
            self.gp_emulator = gp_emulator

            # Choose training data
            train_data, test_data = self.gp_emulator.set_train_test_data(
                self.cs_params.sep_fact, self.cs_params.seed
            )

        ##Call bo_iter
        results_df, max_ei_details_df, list_gp_emulator_class, why_term = (
            self.__run_bo_to_term(run_num, job)
        )

        # # Set results for all compiled iterations for that run
        bo_results_res, bo_results_GPs = self.__make_BO_results_temp(results_df, why_term, max_ei_details_df, list_gp_emulator_class)

        # return bo_results_res, bo_results_GPs
        return bo_results_res, bo_results_GPs
    
    def reset_rng(self):
        """
        Resets the random number generator to the seed value
        """
        if self.cs_params.seed is not None:
            self.rng_set = np.random.default_rng(self.cs_params.seed)
        if self.simulator.sim_seed is not None:
            self.simulator.rng_set = np.random.default_rng(self.simulator.sim_seed)
    
    def run_bo_restarts(self, job = None):
        """
        Runs multiple GPBO restarts

        Returns
        --------
        gpbo_res_simple: list(BO_Results)
            Includes the most relevant results related to a set of BO iters for all restarts
        gpbo_res_GP: list(BO_Results)
            Includes the GP emulator classes used and max ei details for each iteration of the BO workflow

        Notes
        ------
        gpbo_res_simple includes the Configuration, Simulator class, Experiment Data Results DataFrame, and termination criteria results
        """
        gpbo_res_simple = []
        gpbo_res_GP = []
        run_start = 0

        if job is not None:
            #Check for files in job for gpbo_res_simple and gpbo_res_GP
            # if os.path.exists("BO_Results.gz") and os.path.exists("BO_Results_GPs.gz"):
            if job.isfile("BO_Results.gz") and job.isfile("BO_Results_GPs.gz"):
                #Load the data from the files
                fileObj1 = gzip.open(job.fn("BO_Results.gz"), "rb")
                # fileObj1 = gzip.open("BO_Results.gz", "rb")
                gpbo_res_simple = pickle.load(fileObj1)
                fileObj1.close()
                fileObj2 = gzip.open(job.fn("BO_Results_GPs.gz"), "rb")
                # fileObj2 = gzip.open("BO_Results_GPs.gz", "rb")
                gpbo_res_GP = pickle.load(fileObj2)
                fileObj2.close()

        self.gpbo_res_simple = gpbo_res_simple
        self.gpbo_res_GP = gpbo_res_GP
        
        simulator_class = self.simulator
        configuration = {
            "DateTime String": self.cs_params.DateTime,
            "Method Name Enum Value": self.method.method_name.value,
            "Case Study Name": self.cs_params.cs_name,
            "Number of Parameters": len(self.simulator.theta_true_names),
            "Number of State Points": self.exp_data.get_num_x_vals(),
            "Exploration Bias Method Value": self.ep_bias.ep_enum.value,
            "Separation Factor": self.cs_params.sep_fact,
            "Normalize": self.cs_params.normalize,
            "Initial Kernel": self.cs_params.kernel,
            "Initial Lengthscale": self.cs_params.lenscl,
            "Initial Outputscale": self.cs_params.outputscl,
            "Retrain GP": self.cs_params.retrain_GP,
            "Reoptimize Obj": self.cs_params.reoptimize_obj,
            "Heat Map Points Generated": self.cs_params.gen_heat_map_data,
            "Max BO Iters": self.cs_params.bo_iter_tot,
            "Number of Workflow Restarts": self.cs_params.bo_run_tot,
            "Seed": self.cs_params.seed,
            "Acq Tolerance": self.cs_params.acq_tol,
            "MC SG Max Points": self.sg_mc_samples,
            "Obj Improvement Tolerance": self.cs_params.obj_tol,
            "Theta Generation Enum Value": self.gen_meth_theta.value,
            "Gen y with Noise": self.cs_params.w_noise,
            "Gen y for Minimized SSE": self.cs_params.get_y_sse,
        }

        #If some runs have already been completed
        if len(self.gpbo_res_simple) > 0:
            # Check if all of the iterations of that runs have been completed 
            if len(self.gpbo_res_GP[-1].list_gp_emulator_class) < self.cs_params.bo_iter_tot:
                #If not, complete the last run before continuing
                run_start = len(gpbo_res_simple) -1
            else:
                #If the run is complete, start from the next run
                run_start = len(gpbo_res_simple)

        #Get the seed based on the run number
        if self.cs_params.seed is not None:
            self.cs_params.seed += run_start

        #Complete remaining runs
        for i in range(run_start, self.cs_params.bo_run_tot, 1):
            #Run the bo workflow and get the results
            bo_results_res, bo_results_GPs = self.__run_bo_workflow(i, job)

            # Update the seed in configuration
            configuration["Seed"] = self.cs_params.seed
            # Add this copy of configuration with the new seed to the bo_results
            bo_results_res.configuration = configuration.copy()
            # # Add simulator class after rng changes (allows us to restart from the next run)
            bo_results_res.simulator_class = copy.deepcopy(simulator_class)
            # On the 1st iteration of the first run, create heat map data if we are actually generating the data
            if i == 0:
                if self.cs_params.gen_heat_map_data == True:
                    # Generate heat map data for each combination of parameter values stored in a dictionary
                    heat_map_data_dict = self.create_heat_map_param_data()
                    # Save these heat map values in the bo_results object
                    # Only store in first list entry to avoid repeated data which stays the same for each iteration.
                    bo_results_GPs.heat_map_data_dict = heat_map_data_dict

            #Save the results to the gpbo_res_simple and gpbo_res_GP lists
            self.gpbo_res_simple[i] = bo_results_res
            self.gpbo_res_GP[i] = bo_results_GPs

            # #At each restart, resave gpbo_res_simple and gpbo_res_GP to the data file
            if job is not None:
                self.save_results_run(job)

            # Add 1 to the seed to get different seeds when the seeds are set at each restart
            if self.cs_params.seed is not None:
                self.cs_params.seed += 1

        return self.gpbo_res_simple, self.gpbo_res_GP

    def save_results_run(self, job):
        """
        Defines where to save data to and saves data accordingly

        Parameters
        ----------
        restart_bo_results: list of class instances of BO_results, The results of all restarts of the BO workflow for reproduction
        """
        ##Define a path for the data. (Use the name of the case study and date)
        #Get Date only from DateTime String
        savepath1 = job.fn("BO_Results.gz")
        # savepath1 = "BO_Results.gz"
        fileObj1 = gzip.open(savepath1, "wb", compresslevel=1)
        pickled_results1 = pickle.dump(self.gpbo_res_simple, fileObj1)
        fileObj1.close()

        savepath2 = job.fn("BO_Results_GPs.gz")
        # savepath2 = "BO_Results_GPs.gz"
        fileObj2 = gzip.open(savepath2, "wb", compresslevel=2)
        pickled_results2 = pickle.dump(self.gpbo_res_GP, fileObj2)
        fileObj2.close()