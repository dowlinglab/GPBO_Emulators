import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import signac

import pytest
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import *  # Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import *  # Fix this later
from bo_methods_lib.bo_methods_lib.analyze_data import *

# FIX SYSTEM PATH PROBLEM
# Get the parent directory
print(os.getcwd())


def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules


criteria_dict = {
    "cs_name_val": 1,
    "gp_package": "gpflow",
    "meth_name_val": {"$in": [1, 2, 3, 4, 5, 6, 7]},
}
project = signac.get_project()
save_csv = False

# This test function tests whether least_squares_analysis works correctly
# tot_runs, expected runs
ls_analysis_list = [[None, 5], [3, 3]]


@pytest.mark.parametrize("tot_runs, exp_runs", ls_analysis_list)
def test_least_squares_analysis(tot_runs, exp_runs):
    ls_analyzer = LS_Analysis(criteria_dict, project, save_csv)
    ls_results = ls_analyzer.least_squares_analysis(tot_runs)
    assert ls_results.shape[1] == 12
    assert max(ls_results["Run"]) == exp_runs


# This test function tests whether least_squares_analysis throws correct errors
# tot_runs
ls_analysis_err_list = [["None", 0]]


@pytest.mark.parametrize("tot_runs", ls_analysis_err_list)
def test_least_squares_analysis_err(tot_runs):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        ls_analyzer = LS_Analysis(criteria_dict, project, save_csv)
        ls_results = ls_analyzer.least_squares_analysis(tot_runs)


# This test function tests whether categ_min works correctly
# tot_runs
categ_min_list = [
    None,
    5,
]  # Note: Since this is exp_data, number of thetas is defined by num_x_data**dim_x


@pytest.mark.parametrize("tot_runs", categ_min_list)
def test_categ_min(tot_runs):
    ls_analyzer = LS_Analysis(criteria_dict, project, save_csv)
    local_mins = ls_analyzer.categ_min(tot_runs)
    assert local_mins.shape[1] == 4
    assert local_mins.shape[0] == 1
    assert (local_mins["Min Obj Cum."] < 1e-5).all()
    assert (local_mins["Optimality"] < 1e-7).all()
    assert (local_mins["Termination"] == 1).all()


# This test function tests whether categ_min throws correct errors
# tot_runs
categ_min_err_list = [["None", 0]]


@pytest.mark.parametrize("tot_runs", categ_min_err_list)
def test_categ_min_err(tot_runs):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        ls_analyzer = LS_Analysis(criteria_dict, project, save_csv)
        local_mins = ls_analyzer.categ_min(tot_runs)
