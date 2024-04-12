import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import signac

import pytest
from bo_methods_lib.bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.bo_methods_lib.GPBO_Class_fxns import * #Fix this later
from bo_methods_lib.bo_methods_lib.analyze_data import *

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
criteria_dict3 = {"cs_name_val" : 13,
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
make_dir_name_list = [[criteria_dict, False, 'Results/cs_name_val_1/gp_package_gpflow/meth_name_val_in_1_2_3_4_5_6_7'],
                    [criteria_dict, True, 'cs_name_val_1/gp_package_gpflow/meth_name_val_in_1_2_3_4_5_6_7'],
                    [criteria_dict2, False, 'Results/cs_name_val_1/gp_package_gpflow/meth_name_val_1'],
                    [criteria_dict2, True, 'cs_name_val_1/gp_package_gpflow/meth_name_val_1']]
                                       
@pytest.mark.parametrize("crit_dict, is_nested, exp_path", make_dir_name_list)
def test_make_dir_name_from_criteria(crit_dict, is_nested, exp_path):
    analyzer = General_Analysis(criteria_dict, project, save_csv)
    dir_name = analyzer.make_dir_name_from_criteria(crit_dict, is_nested)
    assert dir_name == exp_path

#This test function tests whether make_dir_name_from_criteria throws correct errors
                    #crit_dict, is_nested
make_dir_name_err_list = [["criteria_dict", True],
                          [criteria_dict_fail, False],
                          [[1,2,3], True],
                          [criteria_dict, "None"],
                          [criteria_dict, 9]]
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

#This test function tests whether get_best_data/get_median_data/get_mean_data work correctly
get_bmm_data_list = [[criteria_dict, 7],
                    [criteria_dict2, 1]]
@pytest.mark.parametrize("crit_dict, exp_jobs", get_bmm_data_list)
def test_get_bmm_data(crit_dict, exp_jobs):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    fxns = [analyzer.get_best_data(), analyzer.get_median_data(), analyzer.get_mean_data()]
    for fxn in fxns: 
        df_best, job_list_best = fxn
        assert len(job_list_best) == exp_jobs
        assert df_best.shape[0] == exp_jobs
        assert df_best.shape[1] == 22

#This test function tests whether analyze_obj_vals works correctly
analyze_obj_vals_list = [[criteria_dict, ["sse"]],
                               [criteria_dict, "min_sse"], 
                               [criteria_dict, ["sse", "min_sse"]],
                               [criteria_dict2, "acq"],
                               [criteria_dict3, "sse"]]
@pytest.mark.parametrize("crit_dict, z_choices", analyze_obj_vals_list)
def test_analyze_obj_vals(crit_dict, z_choices):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    jobs = analyzer.get_jobs_from_criteria()
    for job in jobs:
        data, data_names, data_true, sp_data = analyzer.analyze_obj_vals(job, z_choices)
        assert isinstance(sp_data, dict)
        assert isinstance(data, np.ndarray)
        assert isinstance(data_names, list)
        assert isinstance(data_true, dict)
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        assert data.shape[2] == len(z_choices)
        assert data.shape[0] == min(sp_data["bo_run_tot"] - sp_data["bo_run_num"] + 1, sp_data["bo_runs_in_job"])
        assert data.shape[1] == sp_data["bo_iter_tot"]
        assert len(data_names) == len(z_choices)
        if not data_true:
            assert len(data_true) == len(z_choices)
            for z_choice in z_choices:
                assert data_true[z_choice].shape[1] == 12

analyze_obj_vals_err_list = [[criteria_dict, "se", 'test'],
                             [criteria_dict, ["min_sse", "se"], 'test'],
                             [criteria_dict, "min_sse_sse", 'test'],
                             [criteria_dict, "sse", "jobs"],
                             [criteria_dict, "sse",None]]
@pytest.mark.parametrize("crit_dict, z_choices, job_val", analyze_obj_vals_err_list)
def test_analyze_obj_vals_err(crit_dict, z_choices, job_val):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    if job_val == "test":
        jobs = analyzer.get_jobs_from_criteria()
    else:
        jobs = job_val
    with pytest.raises((AssertionError, AttributeError, ValueError, TypeError)): 
        for job in jobs:
            data, data_names, data_true, sp_data = analyzer.analyze_obj_vals(job, z_choices)

#This test function tests whether analyze_thetas works correctly
analyze_thetas_list = [[criteria_dict, "min_sse"], 
                        [criteria_dict2, "acq"],
                        [criteria_dict3, "sse"]]
@pytest.mark.parametrize("crit_dict, z_choices", analyze_thetas_list)
def test_analyze_thetas(crit_dict, z_choices):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    jobs = analyzer.get_jobs_from_criteria()
    for job in jobs:
        data, data_names, data_true, sp_data = analyzer.analyze_thetas(job, z_choices)
        assert isinstance(sp_data, dict)
        assert isinstance(data, np.ndarray)
        assert isinstance(data_names, list)
        assert isinstance(data_true, dict)
        assert data.shape[2] == len(data_true)
        assert data.shape[0] == min(sp_data["bo_run_tot"] - sp_data["bo_run_num"] + 1, sp_data["bo_runs_in_job"])
        assert data.shape[1] == sp_data["bo_iter_tot"]
        assert len(data_names) == data.shape[2]

analyze_thetas_err_list = [[criteria_dict, "se",'test'],
                           [criteria_dict, ["sse"],'test'],
                             [criteria_dict, ["min_sse", "se"],'test'],
                             [criteria_dict, "min_sse_sse",'test'],
                             [criteria_dict, "sse","jobs"],
                             [criteria_dict, "sse",None]]
@pytest.mark.parametrize("crit_dict, z_choices, job_val", analyze_thetas_err_list)
def test_analyze_thetas_err(crit_dict, z_choices, job_val):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    if job_val == "test":
        jobs = analyzer.get_jobs_from_criteria()
    else:
        jobs = job_val
    with pytest.raises((AssertionError, AttributeError, TypeError)): 
        for job in jobs:
            data, data_names, data_true, sp_data = analyzer.analyze_thetas(job, z_choices)

analyze_hypers_list = [criteria_dict, criteria_dict2]
@pytest.mark.parametrize("crit_dict", analyze_hypers_list)
def test_analyze_hypers(crit_dict):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    jobs = analyzer.get_jobs_from_criteria()
    for job in jobs:
        data, data_names, data_true, sp_data = analyzer.analyze_hypers(job)
        assert isinstance(sp_data, dict)
        assert isinstance(data, np.ndarray)
        assert isinstance(data_names, list)
        assert data_true == None
        assert data.shape[2] == len(data_names)
        assert data.shape[0] == min(sp_data["bo_run_tot"] - sp_data["bo_run_num"] + 1, sp_data["bo_runs_in_job"])
        assert data.shape[1] == sp_data["bo_iter_tot"]

#This test function tests whether get_study_data_signac throws correct errors
                    #crit_dict, is_nested
analyze_hypers_err_list = ["jobs", None]
@pytest.mark.parametrize("job_val", analyze_hypers_err_list)
def test_analyze_hypers_err(job_val):
    with pytest.raises((AssertionError, AttributeError, ValueError, TypeError)): 
        analyzer = General_Analysis(criteria_dict, project, save_csv)
        for job in job_val:
            data, data_names, data_true, sp_data = analyzer.analyze_hypers(job)

#This test function tests whether analyze_parity_plot_data works correctly
analyze_pp_data_list = [criteria_dict,criteria_dict2, criteria_dict3]
@pytest.mark.parametrize("crit_dict", analyze_pp_data_list)
def test_analyze_pp_data(crit_dict):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    df_best, job_list_best = analyzer.get_best_data()
    for job, run_num, bo_iter in zip(job_list_best, df_best["Run Number"], df_best["BO Iter"]):
        test_data = analyzer.analyze_parity_plot_data(job, run_num, bo_iter)
        assert isinstance(test_data, Data)
        assert all(value is not None for value in [test_data.x_vals, test_data.y_vals, test_data.theta_vals])

analyze_pp_data_err_list = [["jobs", "test", "test"],
                            [None, "test", "test"],
                            ["jobs","test", "test"],
                            ["test", "run", "test"],
                            ["test", None, "test"],
                            ["test", 10, "test"],
                            ["test", "test", "iter"],
                            ["test", "test", 100],
                            ["test", "test", None]]
@pytest.mark.parametrize("job_val, run, iter", analyze_pp_data_err_list)
def test_analyze_pp_data_err(job_val, run, iter):
    analyzer = General_Analysis(criteria_dict, project, save_csv)
    df_best, jobs = analyzer.get_best_data()
    with pytest.raises((AssertionError, AttributeError, ValueError, TypeError)):
        for job, run_num, bo_iter in zip(jobs, df_best["Run Number"], df_best["BO Iter"]):
            if job_val != "test":
                job = job_val
            if run != "test":
                run_num = run
            if iter != "test":
                bo_iter = iter
            test_data = analyzer.analyze_parity_plot_data(job, run_num, bo_iter)

#This test function tests whether analyze_heat_maps works correctly
analyze_heat_maps_list = [[criteria_dict, True],
                          [criteria_dict, False],
                          [criteria_dict2, True],
                          [criteria_dict3, True]]
@pytest.mark.parametrize("crit_dict, get_ei", analyze_heat_maps_list)
def test_analyze_heat_maps_data(crit_dict, get_ei):
    analyzer = General_Analysis(crit_dict, project, save_csv)
    df_best, job_list_best = analyzer.get_best_data()
    #Back out number of parameters
    string_val = df_best["Theta Min Obj"].iloc[0]
    try:
        numbers = [float(num) for num in string_val.replace('[', '').replace(']', '').split()]
    except:
        numbers = [float(num) for num in string_val]
    #Create list of parameter pair combinations
    dim_theta = len(np.array(numbers).reshape(-1, 1))
    dim_list = np.linspace(0, dim_theta-1, dim_theta)
    pairs = len((list(combinations(dim_list, 2))))
    for job, run_num, bo_iter in zip(job_list_best, df_best["Run Number"], df_best["BO Iter"]):
        for pair_id in range(pairs):
            for job, run_num, bo_iter in zip(job_list_best, df_best["Run Number"], df_best["BO Iter"]):
                all_data, test_mesh, param_info_dict, sp_data = analyze_heat_maps(job, run_num, bo_iter, pair_id, get_ei)
                assert isinstance(all_data, list)
                assert isinstance(test_mesh, np.ndarray)
                assert isinstance(param_info_dict, dict)
                assert isinstance(sp_data, dict)
                assert all(value is not None for value in [all_data, test_mesh, param_info_dict, sp_data])

analyze_hm_err_list = [["jobs", "test", "test", "test", "test"],
                        [None, "test", "test", "test", "test"],
                        ["jobs","test", "test", "test", "test"],
                        ["test", "run", "test", "test", "test"],
                        ["test", None, "test", "test", "test"],
                        ["test", 10, "test", "test", "test"],
                        ["test", "test", "iter", "test", "test"],
                        ["test", "test", 100, "test", "test"],
                        ["test", "test", None, "test", "test"],
                        ["test", "test", "test", "pair_id", "test"],
                        ["test", "test", "test", None, "test"],
                        ["test", "test", "test", 50, "test"],
                        ["test", "test", "test", "test", "get_ei"]]
@pytest.mark.parametrize("job_val, run, iter, pair, ei_val", analyze_hm_err_list)
def test_analyze_hm_err(job_val, run, iter, pair, ei_val):
    analyzer = General_Analysis(criteria_dict, project, save_csv)
    df_best, jobs = analyzer.get_best_data()
    #Back out number of parameters
    string_val = df_best["Theta Min Obj"].iloc[0]
    try:
        numbers = [float(num) for num in string_val.replace('[', '').replace(']', '').split()]
    except:
        numbers = [float(num) for num in string_val]
    #Create list of parameter pair combinations
    dim_theta = len(np.array(numbers).reshape(-1, 1))
    dim_list = np.linspace(0, dim_theta-1, dim_theta)
    pairs = len((list(combinations(dim_list, 2))))
    with pytest.raises((AssertionError, AttributeError, ValueError, TypeError)):
        for pair_id in range(pairs):
            if pair != "test":
                pair_id = pair
            for job, run_num, bo_iter in zip(jobs, df_best["Run Number"], df_best["BO Iter"]):
                if job_val != "test":
                    job = job_val
                if run != "test":
                    run_num = run
                if iter != "test":
                    bo_iter = iter
                if pair != "test":
                    bo_iter = iter
                if ei_val != "test":
                    get_ei = False
                else:
                    get_ei = ei_val
                all_data, test_mesh, param_info_dict, sp_data = analyze_heat_maps(job, run_num, bo_iter, pair_id, get_ei = True)