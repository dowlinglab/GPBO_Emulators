import numpy as np
from scipy.stats import qmc
import pandas as pd
import math
import bo_methods_lib
from .GPBO_Classes_New import Simulator
from pyomo.environ import *

def simulator_helper_test_fxns(cs_num, noise_mean, noise_std, seed):
    """
    Sets the model for calculating y based off of the case study identifier.
    Parameters
    ----------
    cs_num: The number associated with the case study value.
    noise_mean:float, int: The mean of the noise
    noise_std: float, int: The standard deviation of the noise. If None, 5% of mean of Y-exp will be used
    seed: int or None, Determines seed for randomizations. None if seed is random

    Returns
    --------
    Simulator(), Simulator() class object
    """
    assert 1 <= cs_num <=17
    #Get class based on cs number
    if cs_num == 1:
        cs_class = CS1()
    elif 2 <= cs_num <=7:
        cs_class = CSMuller(cs_num)
    elif cs_num == 8:
        cs_class = CS8()
    elif cs_num == 9:
        cs_class = CS9()
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
    else:
        raise(ValueError, "cs_num must be from 1 to 17")
    
    return Simulator(cs_class.idcs_to_consider, 
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
                    cs_class.calc_y_fxn_args)


class CS1:
    def __init__(self):
        self.param_name_str = "t1t2" 
        self.idcs_to_consider = [0,1]
        self.theta_names = ['theta_1', 'theta_2']
        self.bounds_x_l = [-2]
        self.bounds_x_u = [2]
        self.bounds_theta_l = [-2, -2]
        self.bounds_theta_u = [ 2,  2]
        self.theta_ref = np.array([1.0, -1.0])
        self.calc_y_fxn = calc_cs1_polynomial
        self.calc_y_fxn_args = None

def calc_cs1_polynomial(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    
    y_poly =  true_model_coefficients[0]*x + true_model_coefficients[1]*x**2 +x**3
    
    return y_poly
         
class CSMuller:
    
    def __init__(self, cs_number):
        assert 2 <= cs_number <=7 
        self.__set_param_str()
        self.__set_idcs_to_consider()
        self.theta_names = ['x0_1', 'x0_2', 'x0_3', 'x0_4']
        self.theta_names = ['A_1', 'A_2', 'A_3', 'A_4', 'a_1', 'a_2', 'a_3', 'a_4', 'b_1', 'b_2', 'b_3', 'b_4', 'c_1', 
                       'c_2', 'c_3', 'c_4', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'y0_1', 'y0_2', 'y0_3', 'y0_4']
        self.bounds_x_l = [-1.5, -0.5]
        self.bounds_x_u = [1, 2]
        self.bounds_theta_l = [-300,-200,-250, 5,-2,-2,-10, -2, -2,-2,5,-2,-20,-20, -10,-1 ,-2,-2,-2, -2,-2,-2,0,-2]
        self.bounds_theta_u = [-100,  0, -150, 20,2, 2, 0,  2,  2,  2, 15,2, 0,0   , 0,  2, 2,  2, 2, 2 ,2 , 2, 2,2]
        self.theta_ref = np.array([-200,-100,-170,15,-1,-1,-6.5,0.7,0,0,11,0.6,-10,-10,-6.5,0.7,1,0,-0.5,-1,0,0.5,1.5,1])
        self.calc_y_fxn = calc_muller
        self.calc_y_fxn_args = {"min muller": self.__solve_pyomo_Muller_min()}

    def __set_param_str(self):
        if self.cs_number == 2:
            param_name_str = "x0"
        elif self.cs_number == 3:
            param_name_str = "y0"
        elif self.cs_number == 4:
            param_name_str = "x0y0"
        elif self.cs_number == 5:
            param_name_str = "Ax0y0"
        elif self.cs_number == 6:
            param_name_str = "Ax0"
        elif self.cs_number == 7:
            param_name_str = "Ay0"

        self.param_name_str = param_name_str

    def __set_idcs_to_consider(self):
        #Set param_name_str
        indecies_to_consider = []
        all_param_idx = [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23]

        if "A" in self.param_name_str:
            indecies_to_consider += all_param_idx[0:4]
        if "x0" in self.param_name_str:
            indecies_to_consider += all_param_idx[16:20]
        if "y0" in self.param_name_str:
            indecies_to_consider += all_param_idx[20:]

        self.idcs_to_consider = indecies_to_consider
    
    def __solve_pyomo_Muller_min(self, verbose = False):
        """
        Creates and Solves a Pyomo model for the Muller potential
        
        Returns:
        --------
        model.obj(): float, The minimum value of the Muller potential for the given sub problem defined by param_name_str
        """
        #Create Model
        model = ConcreteModel()

        # Create a Set to represent the iterable set of variables A1-A4, b1-b4,...y01-y04
        index_set = range(1,5)
        if "A" in self.param_name_str:
            model.A = Var(Set(initialize=index_set), initialize={1: -210, 2: -100, 3: -200, 4: 10}, 
                    bounds={1: (-300,-100) , 2: (-200,0), 3: (-250,-150), 4: (5,20)})
        else:
            model.A = Param(Set(initialize=index_set), initialize={1: -200, 2: -100, 3: -170, 4: 15})

        model.a = Param(Set(initialize=index_set), initialize={1: -1, 2: -1, 3: -6.5, 4: 0.7})
        model.b = Param(Set(initialize=index_set), initialize={1: 0, 2: 0, 3: 11, 4: 0.6})
        model.c = Param(Set(initialize=index_set), initialize={1: -10, 2: -10, 3: -6.5, 4: 0.7})

        if "x0" in self.param_name_str:
            model.x0 = Var(Set(initialize=index_set), initialize={1: 0, 2: 0, 3: 0, 4: 0}, 
                    bounds={1: (-2,2), 2: (-2,2), 3: (-2,2), 4: (-2,2)})
        else:
            model.x0 = Param(Set(initialize=index_set), initialize={1: 1, 2: 0, 3: -0.5, 4: -1})

        if "y0" in self.param_name_str:
            model.y0 = Var(Set(initialize=index_set), initialize={1: 0, 2: 0, 3: 1, 4: 0}, 
                    bounds={1: (-2,2), 2: (-2,2), 3: (0,2), 4: (-2,2)})
        else:
            model.y0 = Param(Set(initialize=index_set), initialize={1: 0, 2: 0.5, 3: 1.5, 4: 1})

        model.x_index = Set(initialize=range(1,3))
        model.x = Var(model.x_index, initialize={1: -1, 2: 0}, bounds={1: (-1.5,1.0), 2: (-0.5,2)})

        #Define Muller potential
        def calc_muller_pyo(model):  
            #Calculate Muller Potential
            expression = sum( (
                    model.A[i] * exp(
                    model.a[i] * (model.x[1] - model.x0[i]) ** 2 +
                    model.b[i] * (model.x[1] - model.x0[i]) * (model.x[2] - model.y0[i]) +
                    model.c[i] * (model.x[2] - model.y0[i]) ** 2) for i in range(1,5) ) )

            return expression
        
        #Define objective
        model.obj = Objective(rule=calc_muller_pyo, sense = minimize)
        
        solver = SolverFactory('ipopt')
        solver.options['max_iter']= 10000
        result = solver.solve(model, tee = verbose)
        
        if verbose:
            # Access solver status and results
            print("Solver Status:", result.solver.status)
            print("Termination Condition:", result.solver.termination_condition)
            # Print the variable value
            #Print model
            model.pprint()
            
        return model.obj()

def calc_muller(model_coefficients, x, args):
        """
        Caclulates the Muller Potential
        
        Parameters
        ----------
            model_coefficients: ndarray, The array containing the values of Muller constants
            x: ndarray, Values of X
            args: dict, extra arguments to pass to the function.
        
        Returns:
        --------
            y_mul: float, value of Muller potential
        """
        assert "min muller" in list(args.keys())
        
        min_muller = args["min muller"]
        
        #Reshape x to matrix form
        #If array is not 2D, give it shape (len(array), 1)
        if not len(x.shape) > 1:
            x = x.reshape(-1,1)
            
        assert x.shape[0] == 2, "Muller Potential x_data must be 2 dimensional"
        X1, X2 = x #Split x into 2 parts by splitting the rows
        
        #Separate all model parameters into their appropriate pieces
        model_coefficients_reshape = model_coefficients.reshape(6, 4)
            
        #Calculate Muller Potential
        A, a, b, c, x0, y0 = model_coefficients_reshape
        term1 = a*(X1 - x0)**2
        term2 = b*(X1 - x0)*(X2 - y0)
        term3 = c*(X2 - y0)**2
        y_mul = np.sum(A*np.exp(term1 + term2 + term3) )
        y_mul_scl = np.log(max(y_mul - min_muller + 1e-12, 1e-12))
        
        return y_mul_scl

class CS8:
    def __init__(self):
        self.param_name_str = "t1t2t3t4t5" 
        self.idcs_to_consider = [0,1,2,3,4]
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4','theta_5']
        self.bounds_x_l = [-5, -5]
        self.bounds_x_u = [ 5,  5]
        self.bounds_theta_l = [-300,-5.0,-20, -5.0, -20]
        self.bounds_theta_u = [   0, 5.0, 20,  5.0,  20]
        self.theta_ref = np.array([-100, -1.0, 10, -0.1, 10])            
        self.calc_y_fxn = calc_cs8_10_polynomial
        self.calc_y_fxn_args = None

class CS10:
    def __init__(self):
        self.param_name_str = "t1t2t3t4t5" 
        self.idcs_to_consider = [0,1,2,3,4]
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4','theta_5']
        self.bounds_x_l = [-5, -5]
        self.bounds_x_u = [ 5,  5]
        self.bounds_theta_l = [-5,-5,-1, 5, -5]
        self.bounds_theta_u = [ 5, 5, 1, 10,  5]
        self.theta_ref = np.array([1, -2, 0.5, 7, -3])
        self.calc_y_fxn = calc_cs8_10_polynomial
        self.calc_y_fxn_args = None
    
def calc_cs8_10_polynomial(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 5, "true_model_coefficients must be length 5"
    t1, t2, t3, t4, t5 = true_model_coefficients
    
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)

    assert x.shape[0] == 2, "Polynomial x_data must be 2 dimensional"
    x1, x2 = x #Split x into 2 parts by splitting the rows

    y_model =  t1*x1 + t2*x2 + t3*x1*x2 + t4*x1**2 + t5*x2**2
    
    return y_model
    
calc_cs3_polynomial = calc_cs8_10_polynomial


class CS9:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4']
        self.idcs_to_consider = [0,1,2,3]
        self.bounds_x_l = [1, 1]
        self.bounds_x_u = [ 11,  11]
        self.bounds_theta_l = [1, 1e-2, 1, 1e-3]
        self.bounds_theta_u = [100, 1, 500,  1e-1]
        self.theta_ref =  np.array([20,0.2,200,0.02])
        self.calc_y_fxn = calc_cs9_isotherm
        self.calc_y_fxn_args = None

def calc_cs9_isotherm(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)

    assert x.shape[0] == 2, "Isotherm x_data must be 2 dimensional"
    x1, x2 = x #Split x into 2 parts by splitting the rows

    y_model =  (t1*t2*x1)/(1+t2*x1) + (t3*t4*x2)/(1+t4*x2)
    
    return y_model
calc_cs4_isotherm = calc_cs9_isotherm

class CS11:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2']
        self.idcs_to_consider = [0,1]
        self.bounds_x_l = [1]
        self.bounds_x_u = [7]
        self.bounds_theta_l = [10, 0]
        self.bounds_theta_u = [30 , 1]
        self.theta_ref =  np.array([19.143, 0.5311])
        self.calc_y_fxn = calc_cs11_BOD
        self.calc_y_fxn_args = None

def calc_cs11_BOD(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 2, "true_model_coefficients must be length 2"
    t1, t2 = true_model_coefficients
    y_model =  t1*(1-np.exp(-t2*x))
    
    return y_model
    
class CS12:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3']
        self.idcs_to_consider = [0,1,2]
        self.bounds_x_l = [0]
        self.bounds_x_u = [100]
        self.bounds_theta_l = [20, 5, 60]
        self.bounds_theta_u = [ 40, 15, 80]
        self.theta_ref = np.array([30.5, 8.25, 75.1]) 
        self.calc_y_fxn = calc_cs12_yield
        self.calc_y_fxn_args = None

def calc_cs12_yield(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 3, "true_model_coefficients must be length 3"
    t1, t2, t3 = true_model_coefficients
    y_model =  t1*(1- t2*x/(100*(1+t2*x/t3)))
    
    return y_model

class CS13:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4']
        self.idcs_to_consider = [0,1,2,3]
        self.bounds_x_l = [0]
        self.bounds_x_u = [15]
        self.bounds_theta_l = [0, 3, 0.01, 0]
        self.bounds_theta_u = [ 1, 10, 5, 5]
        self.theta_ref = np.array([0.35, 4.54, 2.47, 1.45]) 
        self.calc_y_fxn = calc_cs13_logit
        self.calc_y_fxn_args = None

def calc_cs13_logit(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    y_model =  t1 + (t2-t1)/(1+(x/t3)**t4)
    
    return y_model

class CS14:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4']
        self.idcs_to_consider = [0,1,2,3]
        self.bounds_x_l = [-5, 0]
        self.bounds_x_u = [ 5, 15]
        self.bounds_theta_l = [0, 3, 0.01, 0]
        self.bounds_theta_u = [ 1, 10, 5, 5]
        self.theta_ref = np.array([0.35, 4.54, 2.47, 1.45]) 
        self.calc_y_fxn = calc_cs14_logit2D
        self.calc_y_fxn_args = None

def calc_cs14_logit2D(true_model_coefficients, x, args = None):
    """
    Caclulates the Muller Potential
    
    Parameters
    ----------
        model_coefficients: ndarray, The array containing the values of Muller constants
        x: ndarray, Values of X
        args: dict, extra arguments to pass to the function.
    
    Returns:
    --------
        y_mul: float, value of Muller potential
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)

    assert x.shape[0] == 2, "Isotherm x_data must be 2 dimensional"
    x1, x2 = x #Split x into 2 parts by splitting the rows

    t1, t2, t3, t4 = true_model_coefficients
    y_model =  x1*t1**2 + (t2-t1*x1)/(1+(x2/t3)**t4)
    
    return y_model

class CS15:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5']
        self.idcs_to_consider = [0,1,2,3,4]
        self.bounds_x_l = [-5]
        self.bounds_x_u = [ 2]
        self.bounds_theta_l = [1e-1, 1e-4, -5,  1e-4, -5]
        self.bounds_theta_u = [3,  1, 5, 1, 5]
        self.theta_ref = np.array([2, 0.4, 0.5, 0.3, -3])  
        self.calc_y_fxn = calc_cs15_model
        self.calc_y_fxn_args = None

def calc_cs15_model(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    t1, t2, t3, t4, t5 = true_model_coefficients
    y1 = 1 - (1- t2)*np.exp(-t1*(x-t3)**2)
    y2 = 1 - (1- t4)*np.exp(-t1*(x-t5)**2)
    y_model =  y1*y2
    
    return y_model

class CS16:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4']
        self.idcs_to_consider = [0,1,2,3]
        self.bounds_x_l = [-2*math.pi, -2*math.pi]
        self.bounds_x_u = [ 3*math.pi, 3*math.pi]
        self.bounds_theta_l = [-2, -2, -2, -2]
        self.bounds_theta_u = [ 2, 2, 2, 2]
        self.theta_ref = np.array([0.75, -1, 1.5, -1.0]) 
        self.calc_y_fxn = calc_cs16_trig
        self.calc_y_fxn_args = None

def calc_cs16_trig(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)

    assert x.shape[0] == 2, "Isotherm x_data must be 2 dimensional"
    x1, x2 = x #Split x into 2 parts by splitting the rows

    t1, t2, t3, t4 = true_model_coefficients
    y_model =  np.sin(t1 + t2*x1) + np.cos(t3 + t4*x2)
    
    return y_model


class CS17:
    def __init__(self):
        self.theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4']
        self.idcs_to_consider = [0,1,2,3]
        self.bounds_x_l = [0]
        self.bounds_x_u = [6*math.pi]
        self.bounds_theta_l = [0, -1, 0, -10]
        self.bounds_theta_u = [5,  1e-1, 5,  0]
        self.theta_ref = np.array([3,-0.2,1,-1]) 
        self.calc_y_fxn = calc_cs17_expcos
        self.calc_y_fxn_args = None

def calc_cs17_expcos(true_model_coefficients, x, args = None):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    args: dict, extra arguments to pass to the function. Default None
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    t1, t2, t3, t4 = true_model_coefficients
    y_model =  t1*np.exp(t2*x)*np.cos(t3*x+t4)
    
    return y_model