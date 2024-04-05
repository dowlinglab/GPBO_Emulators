import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import signac

import bo_methods_lib
import pytest
from bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.GPBO_Class_fxns import * #Fix this later
from bo_methods_lib.analyze_data import *

#FIX SYSTEM PATH PROBLEM

def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules

criteria_dict = {"cs_name_val" : 1,
                 "gp_package":"gpflow",
                 "meth_name_val": {"$in": [1,2,3,4,5,6,7]}}
criteria_dict2 = {"cs_name_val" : 1,
                 "gp_package":"gpflow",
                 "meth_name_val": 1}
criteria_dict_fail = {"cs_name_val" : 1,
                 "gp_pack":"gpflow",
                 "meth_name_val": 1}
project = signac.get_project()
save_csv = False
test_analyzer = General_Analysis(criteria_dict, project, save_csv)

#This test function tests whether make_dir_name_from_criteria works correctly
                    #crit_dict, is_nested, exp_path
make_dir_name_list = [[criteria_dict, True, 'Results/cs_name_val_1/ep_enum_val_1/gp_package_gpflow/meth_name_val_in_1_2_3_4_5_6_7'],
                    [criteria_dict, False, 'cs_name_val_1/ep_enum_val_1/gp_package_gpflow/meth_name_val_in_1_2_3_4_5_6_7'],
                    [criteria_dict2, True, 'Results/cs_name_val_1/ep_enum_val_1/gp_package_gpflow/meth_name_val_1'],
                    [criteria_dict2, False, 'cs_name_val_1/ep_enum_val_1/gp_package_gpflow/meth_name_val_1']]
                                       
@pytest.mark.parametrize("crit_dict, is_nested, exp_path", make_dir_name_list)
def test_make_dir_name_from_criteria(crit_dict, is_nested, exp_path):
    analyzer = General_Analysis(criteria_dict, project, save_csv)
    dir_name = analyzer.make_dir_name_from_criteria(crit_dict, is_nested)
    assert dir_name == exp_path

#This test function tests whether make_dir_name_from_criteria throws correct errors
                    #crit_dict, is_nested
make_dir_name_err_list = [["criteria_dict", True],
                          [criteria_dict_fail, True],
                          [[1,2,3], True],
                          [criteria_dict, "None"],
                          [criteria_dict, 0]]
@pytest.mark.parametrize("crit_dict, is_nested", make_dir_name_err_list)
def test_make_dir_name_from_criteria_err(crit_dict, is_nested):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        analyzer = General_Analysis(criteria_dict, project, save_csv)
        dir_name = analyzer.make_dir_name_from_criteria(crit_dict, is_nested)

#This test function tests whether get_jobs_from_criteria works correctly
get_jobs_from_criteria_list = [[criteria_dict, 7],
                               [criteria_dict2, 1]]
@pytest.mark.parametrize("crit_dict, exp_jobs", get_jobs_from_criteria_list)
def test_get_jobs_from_criteria(crit_dict, exp_jobs):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    jobs = analyzer.get_jobs_from_criteria()
    assert len(jobs) == exp_jobs

#This test function tests whether get_df_all_jobs works correctly
get_df_all_jobs_list = [[criteria_dict, 7],
                        [criteria_dict2, 1]]
@pytest.mark.parametrize("crit_dict, exp_jobs", get_df_all_jobs_list)
def test_get_df_all_jobs(crit_dict, exp_jobs):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    df_all_jobs, job_list, theta_true_data = analyzer.get_df_all_jobs()
    assert df_all_jobs.shape[1] == 21
    assert len(job_list) == exp_jobs
    assert isinstance(theta_true_data, dict)
    assert set(theta_true_data.values()) == {1.0,-1.0}

#This test function tests whether get_study_data_signac works correctly
get_study_data_signac_list = [criteria_dict, criteria_dict2]
@pytest.mark.parametrize("crit_dict", get_study_data_signac_list)
def test_get_study_data_signac(crit_dict):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    jobs = analyzer.get_jobs_from_criteria()
    for job in jobs:
        df_job, theta_true_data = analyzer.get_study_data_signac(job)
    assert df_job.shape[1] == 21 and df_job.shape[0] <= 250
    assert set(theta_true_data.values()) == {1.0,-1.0}

#This test function tests whether get_study_data_signac throws correct errors
                    #crit_dict, is_nested
test_jobs = test_analyzer.get_jobs_from_criteria()
get_study_data_signac_err_list = ["job", test_jobs, None]
@pytest.mark.parametrize("job", get_study_data_signac_err_list)
def test_get_study_data_signac_err(job):
    with pytest.raises((AssertionError, AttributeError, ValueError)): 
        analyzer = General_Analysis(criteria_dict, project, save_csv)
        df_job, theta_true_data = analyzer.get_study_data_signac(job)