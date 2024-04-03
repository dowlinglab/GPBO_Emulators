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


def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules
    
#This test function tests whether Enum Classes will call the correct errors
            ##Method number, method name expected, emulator status expected,  obj status expected, sparse grid status expected, MC stat exp.
GPBO_Methods_array = [[Method_name_enum(1), "A1", False, "OBJ", False, False],
                      [Method_name_enum(2), "B1", False, "LN_OBJ", False, False],
                      [Method_name_enum(3), "A2", True, "OBJ", False, False],
                      [Method_name_enum(4), "B2", True, "LN_OBJ", False, False],
                      [Method_name_enum(5), "C2", True, "OBJ", True, False],
                      [Method_name_enum(6), "D2", True, "OBJ", False, True],
                      [Method_name_enum(7), "A3", True, "OBJ", False, False]]
@pytest.mark.parametrize("meth_id, meth_name_e, emulator_e, obj_e, sparse_grid_e, monte_carlo_e", GPBO_Methods_array)
def test_GPBO_Methods(meth_id, meth_name_e, emulator_e, obj_e, sparse_grid_e, monte_carlo_e):
    method = GPBO_Methods(meth_id)
    assert method.method_name.name == meth_name_e and method.emulator == emulator_e and method.obj.name == obj_e and method.sparse_grid == sparse_grid_e and method.mc == monte_carlo_e
    
#This test function tests whether GPBO_Method throws correct errors
                            #meth_id
GPBO_Methods_err_array =   ["C2", None, 3]
@pytest.mark.parametrize("meth_id", GPBO_Methods_err_array)
def test_GPBO_Methods_err(meth_id):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        method = GPBO_Methods(meth_id)