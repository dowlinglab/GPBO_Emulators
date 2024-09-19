import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations

import bo_methods_lib
import pytest
from bo_methods_lib.GPBO_Classes_New import *  # Fix this later
from bo_methods_lib.GPBO_Class_fxns import *  # Fix this later

# Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
# print("Date and Time: ", timestampStr)
# DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M-%S%p")
DateTime = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
DateTime = None  ##For Testing


def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules


cs_name = "Simple Linear"
indecies_to_consider = list(
    range(0, 2)
)  # This is what changes for different subproblems of CS1

ep0 = 1
sep_fact = 0.8
normalize = False
lhs_gen_theta = True
eval_all_pairs = False
noise_mean = 0
noise_std = 0.01
noise_std = 0
kernel = Kernel_enum(1)
set_lenscl = 1
outputscl = False
retrain_GP = 2
GP_train_iter = 300
bo_iter_tot = 3
bo_run_tot = 2
save_fig = False
save_data = False
num_data = None
seed = 1
ei_tol = 1e-6
obj_tol = 1e-4

simulator = simulator_helper_test_fxns(cs_name.value, noise_mean, noise_std, seed)

# How to combine into 1 test function?
# This test function tests whether exp_data is generated in the correct amount
# num_x_data, gen_meth_x, expected number of points generated
gen_exp_data_list = [
    [1, Gen_meth_enum(1), 1],
    [5, Gen_meth_enum(1), 5],
    [1, Gen_meth_enum(2), 1],
    [5, Gen_meth_enum(2), 5],
]


@pytest.mark.parametrize("num_x_data, gen_meth_x, expected", gen_exp_data_list)
def test_gen_exp_data(num_x_data, gen_meth_x, expected):
    exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)
    assert (
        len(exp_data.theta_vals)
        == len(exp_data.x_vals)
        == len(exp_data.y_vals)
        == expected
    ), "y_vals, theta_vals and x_vals should be same length"


# This test function tests whether exp_data will call the correct error
##num_x_data, gen_meth_x, expected number of points generated
gen_exp_data_err_list = [
    [0, Gen_meth_enum(1)],
    [0, Gen_meth_enum(2)],
    [-1, Gen_meth_enum(1)],
    [-1, Gen_meth_enum(2)],
]


@pytest.mark.parametrize("num_x_data, gen_meth_x", gen_exp_data_err_list)
def test_gen_exp_data_err(num_x_data, gen_meth_x):
    with pytest.raises(ValueError):
        exp_data = simulator.gen_exp_data(num_x_data, gen_meth_x)


# This test function tests whether sim_data is generated in the correct amount
# num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected number of data generated
gen_sim_data_list = [
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), 1],
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(2), 1],
    [1, 1, Gen_meth_enum(2), Gen_meth_enum(1), 1],
    [1, 1, Gen_meth_enum(2), Gen_meth_enum(2), 1],
    [5, 5, Gen_meth_enum(1), Gen_meth_enum(1), 25],
    [5, 5, Gen_meth_enum(1), Gen_meth_enum(2), 25],
    [5, 5, Gen_meth_enum(2), Gen_meth_enum(1), 125],
    [5, 5, Gen_meth_enum(2), Gen_meth_enum(2), 125],
    [5, 1, Gen_meth_enum(1), Gen_meth_enum(1), 5],
    [5, 1, Gen_meth_enum(1), Gen_meth_enum(2), 5],
    [5, 1, Gen_meth_enum(2), Gen_meth_enum(1), 25],
    [5, 1, Gen_meth_enum(2), Gen_meth_enum(2), 25],
    [1, 5, Gen_meth_enum(1), Gen_meth_enum(1), 5],
    [1, 5, Gen_meth_enum(1), Gen_meth_enum(2), 5],
    [1, 5, Gen_meth_enum(2), Gen_meth_enum(1), 5],
    [1, 5, Gen_meth_enum(2), Gen_meth_enum(2), 5],
]


@pytest.mark.parametrize(
    "num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected",
    gen_sim_data_list,
)
def test_gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected):
    sim_data = simulator.gen_sim_data(
        num_theta_data, num_x_data, gen_meth_theta, gen_meth_x
    )
    assert (
        len(sim_data.theta_vals) == len(sim_data.y_vals) == expected
    ), "Need same number of theta_vals and y_vals generated"


# This test function tests whether theta data is generated in the correct amount
# num_theta_data, expected number of data generated
gen_theta_vals_list = [[1, 1], [100, 100], [52, 52], [1000, 1000]]


@pytest.mark.parametrize("num_theta_data, expected", gen_theta_vals_list)
def test_gen_theta_vals(num_theta_data, expected):
    sim_data = simulator.gen_theta_vals(num_theta_data)
    assert (
        len(sim_data) == expected
    ), "Need same number of theta_vals and expected values generated"


# This test function tests whether sim_data will not generate y data for gen_val_data = True
# num_theta_data, num_x_data, gen_meth_theta, gen_meth_x
gen_sim_data_val_list = [
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), False, False],
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), True, True],
]


@pytest.mark.parametrize(
    "num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, gen_val, y_generated",
    gen_sim_data_val_list,
)
def test_gen_sim_data_val(
    num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, gen_val, y_generated
):
    sim_data = simulator.gen_sim_data(
        num_theta_data,
        num_x_data,
        gen_meth_theta,
        gen_meth_x,
        sep_fact,
        gen_val_data=gen_val,
    )
    y_vals_gen = sim_data.y_vals == None
    assert y_vals_gen == y_generated


# This test function tests whether sim_data will correctly will throw the correct errors
# num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, set_seed, gen_val
gen_sim_data_err_list = [
    [0, 0, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [0, 1, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [1, 0, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [-1, 1, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [1, -1, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [-1, 0, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [0, -1, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, False],
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, 1.5, False],
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), sep_fact, seed, "string"],
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), 0, seed, False],
]


@pytest.mark.parametrize(
    "num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, seed, gen_val",
    gen_sim_data_err_list,
)
def test_gen_sim_data_err(
    num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, seed, gen_val
):
    with pytest.raises((AssertionError, ValueError)):
        sim_data = simulator.gen_sim_data(
            num_theta_data,
            num_x_data,
            gen_meth_theta,
            gen_meth_x,
            sep_fact,
            seed,
            gen_val,
        )


# This test function tests whether sim_data will correctly will throw the correct errors
# num_theta_data
gen_theta_vals_err_list = [0, -1, None, "1"]


@pytest.mark.parametrize("num_theta_data", gen_theta_vals_err_list)
def test_gen_theta_vals_err(num_theta_data):
    with pytest.raises((AssertionError, ValueError)):
        sim_data = simulator.gen_theta_vals(num_theta_data)


# This test function tests whether y_data is generated correctly for CS1
# num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected y_data generated
gen_y_data_list = [
    [1, 1, Gen_meth_enum(2), Gen_meth_enum(2), [-12]],
    [2, 2, Gen_meth_enum(2), Gen_meth_enum(2), [-12, -4, 4, 12, -20, 4, -4, 20]],
    [1, 2, Gen_meth_enum(2), Gen_meth_enum(2), [-12, -4]],
    [2, 1, Gen_meth_enum(2), Gen_meth_enum(2), [-12, 4, -20, -4]],
]


@pytest.mark.parametrize(
    "num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected", gen_y_data_list
)
def test_gen_y_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected):
    # Generate feature data
    data = simulator.gen_sim_data(
        num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
    )
    # generate y_data
    y_data = simulator.gen_y_data(data, 0, 0)
    assert np.allclose(
        y_data, expected
    ), "Check that y_data is outputting the correct values"


# This test function tests whether sim_data_to_sse_sim_data generates the correct value and length of array
# Generate a test sim_data and exp_data class
sim_data = simulator.gen_sim_data(5, 2, Gen_meth_enum(2), Gen_meth_enum(2), sep_fact)
exp_data = simulator.gen_exp_data(2, Gen_meth_enum(2))
# method, sim_data, exp_data, expected array where best point is found, expected value of that point
sim_data_to_sse_sim_data_list = [
    [GPBO_Methods(Method_name_enum(1)), sim_data, exp_data, [1, -1], 0],
    [GPBO_Methods(Method_name_enum(2)), sim_data, exp_data, [1, -1], -np.inf],
]


@pytest.mark.parametrize(
    "method, sim_data, exp_data, expected_arr, expected_val",
    sim_data_to_sse_sim_data_list,
)
def test_sim_data_to_sse_sim_data(
    method, sim_data, exp_data, expected_arr, expected_val
):
    sse_sim_data = simulator.sim_data_to_sse_sim_data(
        method, sim_data, exp_data, sep_fact
    )
    assert np.allclose(
        sse_sim_data.theta_vals[np.argmin(sse_sim_data.y_vals)], expected_arr, atol=0.01
    )
    assert np.isclose(np.min(sse_sim_data.y_vals), expected_val, 0.01)


# This test function tests whether sim_data_to_sse_sim_data correctly does not generate y for gen_val_data = True
# Generate a test sim_data and exp_data class
# method, sim_data, exp_data, expected array where best point is found, expected value of that point
sim_data_to_sse_sim_data_val_list = [
    [GPBO_Methods(Method_name_enum(1)), sim_data, exp_data, False, False],
    [GPBO_Methods(Method_name_enum(2)), sim_data, exp_data, True, True],
]


@pytest.mark.parametrize(
    "method, sim_data, exp_data, gen_val_data, y_generated",
    sim_data_to_sse_sim_data_val_list,
)
def test_sim_data_to_sse_sim_val_data(
    method, sim_data, exp_data, gen_val_data, y_generated
):
    sim_data = simulator.gen_sim_data(
        1, 1, Gen_meth_enum(2), Gen_meth_enum(2), sep_fact, gen_val_data
    )
    exp_data = simulator.gen_exp_data(1, Gen_meth_enum(2))
    sse_sim_data = simulator.sim_data_to_sse_sim_data(
        method, sim_data, exp_data, sep_fact, gen_val_data
    )
    y_vals_gen = sse_sim_data.y_vals == None
    assert y_vals_gen == y_generated


## Case Study 2 Tests
cs_name2 = "Muller x0"
indecies_to_consider2 = list(
    range(16, 24)
)  # This is what changes for different subproblems of CS2
simulator2 = simulator_helper_test_fxns(cs_name2.value, noise_mean, noise_std, seed)

# This test function tests whether exp_data is generated in the correct amount
# num_x_data, gen_meth_x, expected number of points generated
gen_exp_data_list2 = [
    [1, Gen_meth_enum(1), 1],
    [2, Gen_meth_enum(1), 2],
    [1, Gen_meth_enum(2), 1],
    [2, Gen_meth_enum(2), 4],
]


@pytest.mark.parametrize("num_x_data, gen_meth_x, expected", gen_exp_data_list2)
def test_gen_exp_data(num_x_data, gen_meth_x, expected):
    exp_data = simulator2.gen_exp_data(num_x_data, gen_meth_x)
    assert (
        len(exp_data.theta_vals)
        == len(exp_data.x_vals)
        == len(exp_data.y_vals)
        == expected
    ), "y_vals, theta_vals and x_vals should be same length"


# This test function tests whether sim_data is generated in the correct amount
# num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected number of data generated
gen_sim_data_list2 = [
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(1), 1],
    [1, 1, Gen_meth_enum(1), Gen_meth_enum(2), 1],
    [1, 1, Gen_meth_enum(2), Gen_meth_enum(1), 1],
    [1, 1, Gen_meth_enum(2), Gen_meth_enum(2), 1],
    [2, 2, Gen_meth_enum(1), Gen_meth_enum(1), 4],
    [2, 2, Gen_meth_enum(1), Gen_meth_enum(2), 8],
    [2, 2, Gen_meth_enum(2), Gen_meth_enum(1), 32],
    [2, 2, Gen_meth_enum(2), Gen_meth_enum(2), 64],
    [2, 1, Gen_meth_enum(1), Gen_meth_enum(1), 2],
    [2, 1, Gen_meth_enum(1), Gen_meth_enum(2), 2],
    [2, 1, Gen_meth_enum(2), Gen_meth_enum(1), 16],
    [2, 1, Gen_meth_enum(2), Gen_meth_enum(2), 16],
    [1, 2, Gen_meth_enum(1), Gen_meth_enum(1), 2],
    [1, 2, Gen_meth_enum(1), Gen_meth_enum(2), 4],
    [1, 2, Gen_meth_enum(2), Gen_meth_enum(1), 2],
    [1, 2, Gen_meth_enum(2), Gen_meth_enum(2), 4],
]


@pytest.mark.parametrize(
    "num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected",
    gen_sim_data_list2,
)
def test_gen_sim_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected):
    sim_data = simulator2.gen_sim_data(
        num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
    )
    assert (
        len(sim_data.theta_vals) == len(sim_data.y_vals) == expected
    ), "Need same number of theta_vals and y_vals generated"


# This test function tests whether y_data is generated correctly for CS2
# num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected y_data generated
gen_y_data_list2 = [
    [1, 1, Gen_meth_enum(2), Gen_meth_enum(2), [5.43394415]],
    [
        1,
        2,
        Gen_meth_enum(2),
        Gen_meth_enum(2),
        [5.43394415, 4.88478833, 7.95111829, 11.50992558],
    ],
]


@pytest.mark.parametrize(
    "num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected", gen_y_data_list2
)
def test_gen_y_data(num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, expected):
    # Generate feature data
    data = simulator2.gen_sim_data(
        num_theta_data, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
    )
    # generate y_data
    y_data = simulator2.gen_y_data(data, 0, 0)
    assert np.allclose(
        y_data, expected, atol=0.1
    ), "Check that y_data is outputting the correct values"
