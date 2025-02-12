import numpy as np
from scipy.stats import qmc
import pandas as pd
import math

# import bo_methods_lib
from .GPBO_Classes_New import Simulator
from pyomo.environ import *


def get_cs_class_from_val(cs_num):
    """
    Returns the class associated with the case study value.
    """
    assert cs_num in [
        1,
        2,
        3,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ], "cs_num must be 1,2,3,10,11,12,13,14,15,16,17 not {}".format(cs_num)
    # Get class based on cs number
    if cs_num == 1:
        cs_class = CS1()
    elif 2 <= cs_num <= 3:
        cs_class = CSMuller(cs_num)
    elif cs_num == 10:
        cs_class = CS10()
    elif cs_num == 11:
        cs_class = CS11()
    elif cs_num == 12:
        cs_class = CS12()
    elif cs_num == 13:
        cs_class = CS13()
    elif cs_num == 14:
        cs_class = CS14()
    elif cs_num == 15:
        cs_class = CS15()
    elif cs_num == 16:
        cs_class = CS16()
    elif cs_num == 17:
        cs_class = CS17()

    return cs_class


def simulator_helper_test_fxns(cs_num, noise_mean, noise_std, seed):
    """
    Sets the model for calculating y based off of the case study identifier.
    Parameters
    ----------
    cs_num: int
        The number associated with the case study value.
    noise_mean:float, int
        The mean of the noise
    noise_std: float, int
        The standard deviation of the noise. If None, 5% of mean of Y-exp will be used
    seed: int or None
        Determines seed for randomizations. None if seed is random

    Returns
    --------
    Simulator(): Simulator
        Simulator() class object

    Raises
    ------
    AssertionError
        If any of the required parameters are missing or not of the correct type or value
    """
    assert cs_num in [
        1,
        2,
        3,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ], "cs_num must be 1,2,3,10,11,12,13,14,15,16,17 not {}".format(cs_num)
    # Get class based on cs number
    if cs_num == 1:
        cs_class = CS1()
    elif 2 <= cs_num <= 3:
        cs_class = CSMuller(cs_num)
    elif cs_num == 10:
        cs_class = CS10()
    elif cs_num == 11:
        cs_class = CS11()
    elif cs_num == 12:
        cs_class = CS12()
    elif cs_num == 13:
        cs_class = CS13()
    elif cs_num == 14:
        cs_class = CS14()
    elif cs_num == 15:
        cs_class = CS15()
    elif cs_num == 16:
        cs_class = CS16()
    elif cs_num == 17:
        cs_class = CS17()

    return Simulator(
        cs_class.idcs_to_consider,
        cs_class.theta_ref,
        cs_class.theta_names,
        cs_class.bounds_theta_l,
        cs_class.bounds_x_l,
        cs_class.bounds_theta_u,
        cs_class.bounds_x_u,
        noise_mean,
        noise_std,
        seed,
        cs_class.calc_y_fxn,
        cs_class.calc_y_fxn_args,
    )


class CS1:
    """
    Class containing constants for Simple Linear Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "Simple Linear"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["theta_1", "theta_2"]
        self.bounds_x_l = [-2]
        self.bounds_x_u = [2]
        self.bounds_theta_l = [-2, -2]
        self.bounds_theta_u = [2, 2]
        self.theta_ref = np.array([1.0, -1.0])
        self.calc_y_fxn = calc_cs1_polynomial
        self.calc_y_fxn_args = None


def calc_cs1_polynomial(true_model_coefficients, x, args=None):
    """
    Calculates the value of y for Simple Linear Case Study

    Parameters
    ----------
    true_model_coefficients: np.ndarray
        The array containing the true values of Theta1 and Theta2
    x: np.ndarray
        The list of xs that will be used to generate y
    args: dict, default None
        Extra arguments to pass to the function

    Returns
    --------
    y_poly: np.ndarray
        The noiseless values of y given theta_true and x

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 2
    """

    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"

    y_poly = true_model_coefficients[0] * x + true_model_coefficients[1] * x**2 + x**3

    return y_poly


class CSMuller:
    """
    Class containing constants for The Muller x0 and y0 Case Studies

    Methods:
    --------
    __init__(): Initializes the class
    __set_param_str(): Sets the param_name_str based on the cs_number
    __set_idcs_to_consider(): Sets the idcs_to_consider based on the param_name_str
    __solve_pyomo_Muller_min(): Creates and Solves a Pyomo model for the minimum of the Muller potential
    """

    def __init__(self, cs_number):
        assert 2 <= cs_number <= 3
        self.cs_number = cs_number
        self.__set_param_str()
        self.name = "Muller " + self.param_name_str
        self.__set_idcs_to_consider()
        self.theta_names = ["x0_1", "x0_2", "x0_3", "x0_4"]
        self.theta_names = [
            "A_1",
            "A_2",
            "A_3",
            "A_4",
            "a_1",
            "a_2",
            "a_3",
            "a_4",
            "b_1",
            "b_2",
            "b_3",
            "b_4",
            "c_1",
            "c_2",
            "c_3",
            "c_4",
            "x0_1",
            "x0_2",
            "x0_3",
            "x0_4",
            "y0_1",
            "y0_2",
            "y0_3",
            "y0_4",
        ]
        self.bounds_x_l = [-1.5, -0.5]
        self.bounds_x_u = [1, 2]
        self.bounds_theta_l = [
            -300,
            -200,
            -250,
            5,
            -2,
            -2,
            -10,
            -2,
            -2,
            -2,
            5,
            -2,
            -20,
            -20,
            -10,
            -1,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            0,
            -2,
        ]
        self.bounds_theta_u = [
            -100,
            0,
            -150,
            20,
            2,
            2,
            0,
            2,
            2,
            2,
            15,
            2,
            0,
            0,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
        self.theta_ref = np.array(
            [
                -200,
                -100,
                -170,
                15,
                -1,
                -1,
                -6.5,
                0.7,
                0,
                0,
                11,
                0.6,
                -10,
                -10,
                -6.5,
                0.7,
                1,
                0,
                -0.5,
                -1,
                0,
                0.5,
                1.5,
                1,
            ]
        )
        self.calc_y_fxn = calc_muller
        self.calc_y_fxn_args = {"min muller": self.__solve_pyomo_Muller_min()}

    def __set_param_str(self):
        """
        Sets the param_name_str based on the cs_number"""
        if self.cs_number == 2:
            param_name_str = "x0"
        elif self.cs_number == 3:
            param_name_str = "y0"
        self.param_name_str = param_name_str

    def __set_idcs_to_consider(self):
        """
        Sets the idcs_to_consider based on the param_name_str"""
        # Set param_name_str
        indecies_to_consider = []
        all_param_idx = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ]

        if "A" in self.param_name_str:
            indecies_to_consider += all_param_idx[0:4]
        if "x0" in self.param_name_str:
            indecies_to_consider += all_param_idx[16:20]
        if "y0" in self.param_name_str:
            indecies_to_consider += all_param_idx[20:]

        self.idcs_to_consider = indecies_to_consider

    def __solve_pyomo_Muller_min(self, verbose=False):
        """
        Creates and Solves a Pyomo model for the Muller potential
        Parameters:
        -----------
        verbose: bool
            If True, prints the solver status and termination condition. Default False

        Returns:
        --------
        model.obj(): float
            The minimum value of the Muller potential for the given sub problem defined by param_name_str
        """
        # Create Model
        model = ConcreteModel()

        # Create a Set to represent the iterable set of variables A1-A4, b1-b4,...y01-y04
        index_set = range(1, 5)
        if "A" in self.param_name_str:
            model.A = Var(
                Set(initialize=index_set),
                initialize={1: -210, 2: -100, 3: -200, 4: 10},
                bounds={1: (-300, -100), 2: (-200, 0), 3: (-250, -150), 4: (5, 20)},
            )
        else:
            model.A = Param(
                Set(initialize=index_set), initialize={1: -200, 2: -100, 3: -170, 4: 15}
            )

        model.a = Param(
            Set(initialize=index_set), initialize={1: -1, 2: -1, 3: -6.5, 4: 0.7}
        )
        model.b = Param(
            Set(initialize=index_set), initialize={1: 0, 2: 0, 3: 11, 4: 0.6}
        )
        model.c = Param(
            Set(initialize=index_set), initialize={1: -10, 2: -10, 3: -6.5, 4: 0.7}
        )

        if "x0" in self.param_name_str:
            model.x0 = Var(
                Set(initialize=index_set),
                initialize={1: 0, 2: 0, 3: 0, 4: 0},
                bounds={1: (-2, 2), 2: (-2, 2), 3: (-2, 2), 4: (-2, 2)},
            )
        else:
            model.x0 = Param(
                Set(initialize=index_set), initialize={1: 1, 2: 0, 3: -0.5, 4: -1}
            )

        if "y0" in self.param_name_str:
            model.y0 = Var(
                Set(initialize=index_set),
                initialize={1: 0, 2: 0, 3: 1, 4: 0},
                bounds={1: (-2, 2), 2: (-2, 2), 3: (0, 2), 4: (-2, 2)},
            )
        else:
            model.y0 = Param(
                Set(initialize=index_set), initialize={1: 0, 2: 0.5, 3: 1.5, 4: 1}
            )

        model.x_index = Set(initialize=range(1, 3))
        model.x = Var(
            model.x_index,
            initialize={1: -1, 2: 0},
            bounds={1: (-1.5, 1.0), 2: (-0.5, 2)},
        )

        # Define Muller potential
        def calc_muller_pyo(model):
            # Calculate Muller Potential
            expression = sum(
                (
                    model.A[i]
                    * exp(
                        model.a[i] * (model.x[1] - model.x0[i]) ** 2
                        + model.b[i]
                        * (model.x[1] - model.x0[i])
                        * (model.x[2] - model.y0[i])
                        + model.c[i] * (model.x[2] - model.y0[i]) ** 2
                    )
                    for i in range(1, 5)
                )
            )

            return expression

        # Define objective
        model.obj = Objective(rule=calc_muller_pyo, sense=minimize)

        solver = SolverFactory("ipopt")
        solver.options["max_iter"] = 10000
        result = solver.solve(model, tee=verbose)

        if verbose:
            # Access solver status and results
            print("Solver Status:", result.solver.status)
            print("Termination Condition:", result.solver.termination_condition)
            # Print the variable value
            # Print model
            model.pprint()

        return model.obj()


def calc_muller(model_coefficients, x, args):
    """
    Caclulates the log-scaled and minimum shifted Muller Potential

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the values of Muller constants
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_mul_scl: float
        Value of log scaled and minimum value shifted Muller potential

    Raises
    ------
    AssertionError
        If "min muller" is not in args keys
    """
    assert "min muller" in list(args.keys())

    min_muller = args["min muller"]

    # Reshape x to matrix form
    # If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1, 1)

    assert x.shape[0] == 2, "Muller Potential x_data must be 2 dimensional"
    X1, X2 = x  # Split x into 2 parts by splitting the rows

    # Separate all model parameters into their appropriate pieces
    model_coefficients_reshape = model_coefficients.reshape(6, 4)

    # Calculate Muller Potential
    A, a, b, c, x0, y0 = model_coefficients_reshape
    term1 = a * (X1 - x0) ** 2
    term2 = b * (X1 - x0) * (X2 - y0)
    term3 = c * (X2 - y0) ** 2
    y_mul = np.sum(A * np.exp(term1 + term2 + term3))
    y_mul_scl = np.log(max(y_mul - min_muller + 1e-12, 1e-12))

    return y_mul_scl


class CS10:
    """
    Class containing constants for Large Linear Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.param_name_str = "t1t2t3t4t5"
        self.name = "Large Linear"
        self.idcs_to_consider = [0, 1, 2, 3, 4]
        self.theta_names = ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5"]
        self.bounds_x_l = [-2, -3]
        self.bounds_x_u = [2, 3]
        self.bounds_theta_l = [-5, -5, 0, 5, -5]
        self.bounds_theta_u = [5, 5, 1, 10, -1]
        self.theta_ref = np.array([1, -2, 0.5, 7, -3])
        self.calc_y_fxn = calc_cs8_10_polynomial
        self.calc_y_fxn_args = None


def calc_cs8_10_polynomial(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the Large Linear case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 5
    """
    assert len(true_model_coefficients) == 5, "true_model_coefficients must be length 5"
    t1, t2, t3, t4, t5 = true_model_coefficients

    # If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1, 1)

    assert x.shape[0] == 2, "Polynomial x_data must be 2 dimensional"
    x1, x2 = x  # Split x into 2 parts by splitting the rows

    y_model = t1 * x1 + t2 * x2 + t3 * x1 * x2 + t4 * x1**2 + t5 * x2**2

    return y_model


class CS11:
    """
    Class containing constants for BOD Curve Case Study

    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2"]
        self.name = "BOD Curve"
        self.idcs_to_consider = [0, 1]
        self.bounds_x_l = [1]
        self.bounds_x_u = [7]
        self.bounds_theta_l = [10, 0]
        self.bounds_theta_u = [30, 1]
        self.theta_ref = np.array([19.143, 0.5311])
        self.calc_y_fxn = calc_cs11_BOD
        self.calc_y_fxn_args = None


def calc_cs11_BOD(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the BOD Curve case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 2
    """
    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"
    t1, t2 = true_model_coefficients
    y_model = t1 * (1 - np.exp(-t2 * x))

    return y_model


class CS12:
    """
    Class containing constants for Yield-Loss Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2", "theta_3"]
        self.name = "Yield-Loss"
        self.idcs_to_consider = [0, 1, 2]
        self.bounds_x_l = [0]
        self.bounds_x_u = [100]
        self.bounds_theta_l = [20, 5, 60]
        self.bounds_theta_u = [40, 15, 80]
        self.theta_ref = np.array([30.5, 8.25, 75.1])
        self.calc_y_fxn = calc_cs12_yield
        self.calc_y_fxn_args = None


def calc_cs12_yield(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the Yield-Loss case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 3
    """
    assert len(true_model_coefficients) == 3, "true_model_coefficients must be length 3"
    t1, t2, t3 = true_model_coefficients
    y_model = t1 * (1 - t2 * x / (100 * (1 + t2 * x / t3)))

    return y_model


class CS13:
    """
    Class containing constants for Log Logistic Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2", "theta_3", "theta_4"]
        self.name = "Log Logistic"
        self.idcs_to_consider = [0, 1, 2, 3]
        self.bounds_x_l = [0]
        self.bounds_x_u = [15]
        self.bounds_theta_l = [0, 3, 0.01, 0]
        self.bounds_theta_u = [1, 10, 5, 5]
        self.theta_ref = np.array([0.35, 4.54, 2.47, 1.45])
        self.calc_y_fxn = calc_cs13_logit
        self.calc_y_fxn_args = None


def calc_cs13_logit(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the Log Logistic case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 4
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    y_model = t1 + (t2 - t1) / (1 + (x / t3) ** t4)

    return y_model


class CS14:
    """
    Class containing constants for 2D Log Logistic Case Study
    Methods:
    --------
    __init__(): Initializes the class"""

    def __init__(self):
        self.theta_names = ["theta_1", "theta_2", "theta_3", "theta_4"]
        self.name = "2D Log Logistic"
        self.idcs_to_consider = [0, 1, 2, 3]
        self.bounds_x_l = [-5, 0]
        self.bounds_x_u = [5, 15]
        self.bounds_theta_l = [0, 3, 0.01, 0]
        self.bounds_theta_u = [1, 10, 5, 5]
        self.theta_ref = np.array([0.35, 4.54, 2.47, 1.45])
        self.calc_y_fxn = calc_cs14_logit2D
        self.calc_y_fxn_args = None


def calc_cs14_logit2D(true_model_coefficients, x, args=None):
    """
    Caclulates the simulated y-values for the 2D Log Logistic case study

    Parameters
    ----------
    model_coefficients: np.ndarray
        The array containing the true parameter values
    x: np.ndarray
        Values of X
    args: dict
        Extra arguments to pass to the function.

    Returns:
    --------
    y_model: float
        Value of the model

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 4
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients

    # If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1, 1)

    assert x.shape[0] == 2, "Isotherm x_data must be 2 dimensional"
    x1, x2 = x  # Split x into 2 parts by splitting the rows

    t1, t2, t3, t4 = true_model_coefficients
    y_model = x1 * t1**2 + (t2 - t1 * x1) / (1 + (x2 / t3) ** t4)

    return y_model

class CS15:
    """
    Class containing constants for Simple Linear Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "Simple Multimodal"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["theta_1", "theta_2"]
        self.bounds_x_l = [-2]
        self.bounds_x_u = [1.5]
        self.bounds_theta_l = [-2, -2]
        self.bounds_theta_u = [2, 2]
        self.theta_ref = np.array([-1.5, 0.5 ])
        self.calc_y_fxn = calc_cs15_polynomial
        self.calc_y_fxn_args = None


def calc_cs15_polynomial(true_model_coefficients, x, args=None):
    """
    Calculates the value of y for Simple Linear Case Study

    Parameters
    ----------
    true_model_coefficients: np.ndarray
        The array containing the true values of Theta1 and Theta2
    x: np.ndarray
        The list of xs that will be used to generate y
    args: dict, default None
        Extra arguments to pass to the function

    Returns
    --------
    y_poly: np.ndarray
        The noiseless values of y given theta_true and x

    Raises
    ------
    AssertionError
        If true_model_coefficients is not of length 2
    """

    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"

    y_poly = (true_model_coefficients[0] * x**3 - true_model_coefficients[1] * x**2 + 2*x - 1)**2 + (true_model_coefficients[0] - true_model_coefficients[1])**2 + (x**2 - 1)**2

    return y_poly

class CS16:
    """
    Class containing constants for the Water + Glycerol VLE Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "Water-Glycerol"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["theta_1", "theta_2"]
        self.bounds_x_l = [0]
        self.bounds_x_u = [1]
        self.bounds_theta_l = [-1e3,-1e3]
        self.bounds_theta_u = [1.2e3,1.2e3]
        self.theta_ref = np.array([27.584,-195.9166])
        self.calc_y_fxn = uniquac_model
        self.calc_y_fxn_args =  {"r" :[0.92, 3.5857], #H2O + Glycerol
                                "q" :[1.4, 3.06],
                                "T" : 100+273.15, #K
                                "R" : 1.98721 , #cal/molK
                                "A": [8.07225,7.10850],
                                "B": [1730.63, 1537.78],
                                "C": [233.426, 210.39],
                                "mode": "P"
                                }

class CS17:
    """
    Class containing constants for the Acetonitrile (ACN) + Water VLE Case Study

    Methods:
    --------
    __init__(): Initializes the class
    """

    def __init__(self):
        self.name = "ACN-Water"
        self.param_name_str = "t1t2"
        self.idcs_to_consider = [0, 1]
        self.theta_names = ["theta_1", "theta_2"]
        self.bounds_x_l = [0]
        self.bounds_x_u = [1]
        self.bounds_theta_l = [-1e4,-5e3]
        self.bounds_theta_u = [1e4,1e4]
        self.theta_ref = np.array([436.4803,225.3647])
        self.calc_y_fxn = uniquac_model
        self.calc_y_fxn_args =  {"r" :[1.8701,0.92], #ACN, H2O
                                "q" :[1.7240,1.4],
                                "T" : 50+273.15, #K
                                "R" : 1.98721 , #cal/molK
                                "A": [7.33986,8.07131],
                                "B": [1482.29,1730.63],
                                "C": [250.523,233.426],
                                "mode": "y"
                                } 



def uniquac_model(unknown_params, xP, args):
    """
    Compute activity coefficients using the UNIQUAC model for a binary mixture.

    Parameters:
    unknown_params : np.array
        A vector containing the unknown interaction energy parameters Δu_ij.
    xP : np.array or float
        Mole fractions x1 (x2 is inferred).
    args : dict
        A dictionary containing necessary additional parameters:
        - "r": np.array, volume parameters for components
        - "q": np.array, surface area parameters for components
        - "R": float, gas constant
        - "T": float, temperature
        - "z": float, coordination number (default 10)
        - "A", "B", "C": Antoine equation parameters for vapor pressure

    Returns:
    np.array or float
        Vapor pressure P.
    """
    # Extract parameters
    r = np.array(args["r"])
    q = np.array(args["q"])
    z = args.get("z", 10)
    R = args["R"]
    T = args["T"]
    A, B, C = np.array(args["A"]), np.array(args["B"]), np.array(args["C"])
    mode = args["mode"]
    
    # Precompute constants
    l = (z / 2) * (r - q) - (r - 1)
    tau = np.exp(-unknown_params / (R * T))
    psat = 10 ** (A - B / (C + (T - 273.15)))

    # Ensure xP is at least 1D
    x1 = np.atleast_2d(xP).reshape(-1,1)
    x2 = 1 - x1
    x = np.hstack([x1, x2])

    # Initialize gamma with ones
    gamma = np.ones_like(x, dtype=float)

    # Identify valid indices where both x1 and x2 are nonzero
    valid_mask = (x1.flatten() > 0) & (x2.flatten() > 0)

    if np.any(valid_mask):
        # Apply valid_mask correctly to both dimensions
        valid_x = x[valid_mask, :]  # Shape (M, 2) where M is number of valid rows

        sum_xq = np.dot(valid_x, q)
        sum_xr = np.dot(valid_x, r)

        theta = (valid_x * q) / sum_xq[:, None]
        psi = (valid_x * r) / sum_xr[:, None]

        lngC = (
            np.log(psi / valid_x) + (z / 2) * q * np.log(theta / psi) + psi[:, ::-1] * (l - r * l[::-1] / r[::-1])
        )

        lngR = (
            -q * np.log(theta + theta[:, ::-1] * tau[::-1]) + theta[:, ::-1] * q * (
                tau[::-1] / (theta + theta[:, ::-1] * tau[::-1]) - tau / (theta[:, ::-1] + theta * tau)
            )
        )

        gamma[valid_mask, :] = np.exp(lngC + lngR)
        
    # Handle infinite dilution cases
    if np.any(~valid_mask):
        # Compute gamma at infinite dilution for both components
        gamma_inf = np.zeros(2)

        # term1 = 1- (r[0]/r[1]) +np.log(r[0]/r[1])
        # term2 = -5*q[0]*(1-(r[0]*q[1])/(r[1]*q[0]) + np.log((r[0]*q[1])/(r[1]*q[0])))

        term1 = np.log(r[0]/r[1])
        term2a = 5*np.log((q[0]*r[1])/(q[1]*r[0])) - np.log(tau[1]) + 1 -tau[0]
        term2 = q[0]*term2a
        term3 = l[0]-(r[0]/r[1])*l[1]
        gamma_inf[0] = np.exp(term1 + term2 + term3)

        term1_x2 = np.log(r[1]/r[0])
        term2a_x2 = 5*np.log((q[1]*r[0])/(q[0]*r[1])) - np.log(tau[0]) + 1 -tau[1]
        term2_x2 = q[1]*term2a_x2
        term3_x2 = l[1]-(r[1]/r[0])*l[0]
        gamma_inf[1] = np.exp(term1_x2 + term2_x2 + term3_x2)

    gamma1 = gamma[:, 0]
    gamma2 = gamma[:, 1]

    #Manually alter gamma values at infinite dilution
    if np.any(x2.flatten() == 0):
        gamma2[-1] = gamma_inf[1]
    if np.any(x1.flatten() == 0):
        gamma1[0] = gamma_inf[0]
        

    P = np.sum(x * gamma * psat, axis=1)
    y = x*gamma*psat/P[:, None]
    if mode == "P":
        var = P # Return scalar if input was scalar-like
    elif mode == "gamma":
        var = gamma1 #Return gamma1
    elif mode == "y":
        var = y[:,0] #Return y1

    # return var[0] if var.shape == (1,) else var # Return scalar if input was scalar-like
    return var