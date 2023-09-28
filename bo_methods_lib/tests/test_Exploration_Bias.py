import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations

import bo_methods_lib
import pytest
from bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.GPBO_Class_fxns import * #Fix this later

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
# print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
DateTime = None ##For Testing

def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules

#This test function tests whether set_ep works as intended
               ## ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var, expected
set_ep_list = [[1, None, Ep_enum(1), 0, 100, 1.5, 0.01, False, 0.2, 0.02, 1],
               [3, None, Ep_enum(1), 0, 100, 1.5, 0.01, False, 0.2, 0.02, 3],
               [1, None, Ep_enum(2), 0, 100, 1.5, 0.01, False, 0.2, 0.02, 1],
               [1, None, Ep_enum(2), 1, 100, 1.5, 0.01, False, 0.2, 0.02, 0.9802],
               [1, 1, Ep_enum(3), 0, 100, 1.5, 0.01, False, 10, 0.02, 1.5],
               [1, 1, Ep_enum(3), 0, 100, 1.5, 0.01, True, 10, 0.02, 0.6667],
               [1, None, Ep_enum(3), 0, 100, 1.5, 0.01, True, 10, 0.02, 1],
               [1, None, Ep_enum(4), 0, 100, 1.5, 0.01, False, 10, 0.02, 0.002]]
@pytest.mark.parametrize("ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var, expected", set_ep_list)
def test_set_ep_list(ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var, expected):
    ep_bias = Exploration_Bias(ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var)
    if ep_enum.value >= 3:
        ep_bias.ep_max = best_error
    ep_bias.set_ep()
    assert np.isclose(ep_bias.ep_curr, expected, atol = 1e-2)
    
#This test function tests whether set_ep throws the correct errors
               ## ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var
set_ep_err_list = [[1, None, "Constant", None, None, None, None, None, None, None],
               [3, None, Ep_enum(1), "iter 1", None, None, None, None, None, None],
               [1, None, Ep_enum(2), 1.1, 100, None, None, None, None, None],
               [None, None, Ep_enum(2), 0, 100, None, 0.01, None, None, None],
               [1, None, Ep_enum(2), 1, 1, None, 0.01, None, None, None],
               [1, None, Ep_enum(2), 1, None, None, 0.01, None, None, None],
               [1, None, Ep_enum(2), None, 1, None, 0.01, None, None, None],
               [None, None, Ep_enum(3), 0, 100, 1.5, None, False, None, None],
               [1, None, Ep_enum(3), 0, 100, None, None, False, None, None],
               [None, None, None, None, None, None, None, None, 0.2, 0.02],
               [None, None, Ep_enum(4), None, None, None, None, None, None, 0.02],
               [None, None, Ep_enum(4), None, None, None, None, None, 0.2, None]]
@pytest.mark.parametrize("ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var", set_ep_err_list)
def test_set_ep_err_list(ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var):
    with pytest.raises((AssertionError, ValueError)):   
        ep_bias = Exploration_Bias(ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f, improvement, best_error, mean_of_var)
        ep_bias.set_ep()