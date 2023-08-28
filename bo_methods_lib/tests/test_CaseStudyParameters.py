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
DateTime_str = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
DateTime_none = None ##For Testing

def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules
    
#This test function tests whether CaseStudyParameters will call the correct errors
        ##cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime, seed, ei_tol, obj_tol
case1 = [CS_name_enum(1), "string_ep0", 0.8, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case2 = [CS_name_enum(1), 1, -1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case3 = [CS_name_enum(1), 1, 1.01, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case4 = [CS_name_enum(1), 1, 1, 1, False, Kernel_enum(1), None, None, 0, 0, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case5 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, 1, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case6 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1.1, 2, False, DateTime_str, 1, 1, 1e-4]
case7 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, -1, 2, False, DateTime_str, 1, 1, 1e-4]
case8 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 0, 2, False, DateTime_str, 1, 1, 1e-4]
case9 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2.1, False, DateTime_str, 1, 1, 1e-4]
case10 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, -2, False, DateTime_str, 1, 1, 1e-4]
case11 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 0, False, DateTime_str, 1, 1, 1e-4]
case12 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, "string", DateTime_str, 1, 1, 1e-4]
case13 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, dateTimeObj, 1, 1, 1e-4]
case14 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_none, 1.1, 1, 1e-4]
case15 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_none, -1, 1, 1e-4]
case16 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_none, 1, "string", 1e-4]
case17 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_none, 1, -1, 1e-4]
case18 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_none, 1, 1e-6, "string"]
case19 = [CS_name_enum(1), 1, 1, False, Kernel_enum(1), None, None, 0, 0, False, 1, 2, False, DateTime_none, 1, 1e-6, -1]
case20 = [CS_name_enum(1), 1, 0.8, False, "MAT_52", None, None, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case21 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), 0, None, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case22 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), None, 0, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case23 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), None, None, -1, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case24 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), None, None, 0, -1, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case25 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), None, None, 0, "str", False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case26 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), None, None, "str", 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case27 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), None, "str", 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]
case28 = [CS_name_enum(1), 1, 0.8, False, Kernel_enum(1), "str", None, 0, 0, False, 1, 2, False, DateTime_str, 1, 1, 1e-4]

CaseStudyParameters_array = [case1, case2, case3, case4, case5, case6, case7, case8, case9, case10, case11, case12, case13, case14, 
                             case15,case16, case17, case18, case19, case20, case21, case22, case23, case24, case25, case26, case27, case28]
@pytest.mark.parametrize("cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime, seed, ei_tol", CaseStudyParameters_array)
def test_CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime, seed, ei_tol):
    with pytest.raises((AssertionError, ValueError)):
        cs_params = CaseStudyParameters(cs_name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP, reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_fig, save_data, DateTime, seed, ei_tol)   
    