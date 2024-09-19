import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations
import copy

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


cs_name1 = "Simple Linear"
cs_name2 = "Muller x0"

num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data1 = 5
num_theta_data2 = 2
gen_meth_theta = Gen_meth_enum(2)

ep0 = 1
sep_fact = 0.8
normalize = True
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 2
seed = 1
method = GPBO_Methods(Method_name_enum(1))  # 1A

# Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1.value, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(
    method, sim_data1, exp_data1, sep_fact
)
val_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(
    method, val_data1, exp_data1, sep_fact, True
)
gp_emulator1_s = Type_1_GP_Emulator(
    sim_sse_data1,
    val_sse_data1,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)
gp_emulator1_e = Type_2_GP_Emulator(
    sim_data1,
    val_data1,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)

# Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2.value, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(
    num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(
    method, sim_data2, exp_data2, sep_fact
)
val_data2 = simulator2.gen_sim_data(
    num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(
    method, val_data2, exp_data2, sep_fact, True
)
gp_emulator2_s = Type_1_GP_Emulator(
    sim_sse_data2,
    val_sse_data2,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)
gp_emulator2_e = Type_2_GP_Emulator(
    sim_data2,
    val_data2,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)

# This test function tests whether get_num_gp_data checker works correctly
# emulator class, expected value
get_num_gp_data_list = [
    [gp_emulator1_s, 25],
    [gp_emulator1_e, 125],
    [gp_emulator2_s, 16],
    [gp_emulator2_e, 400],
]


@pytest.mark.parametrize("gp_emulator, expected", get_num_gp_data_list)
def test_get_num_gp_data(gp_emulator, expected):
    assert gp_emulator.get_num_gp_data() == expected


# This test function tests whether bounded_parameter throws correct errors
# lb, ub, init_value
bounded_parameter_err_list = [[2, 4, 3]]


@pytest.mark.parametrize("lb, ub, init_value", bounded_parameter_err_list)
def bounded_parameter_err(lb, ub, init_value):
    bound_param = gp_emulator1_s.bounded_parameter(lb, ub, init_value)
    assert isinstance(bound_param, gpflow.Parameter)


# This test function tests whether bounded_parameter throws correct errors
# lb, ub, init_value
bounded_parameter_err_list = [
    [4, 2, 3],
    ["a", 2, 3],
    [2, "a", 3],
    [2, 3, "a"],
]


@pytest.mark.parametrize("lb, ub, init_value", bounded_parameter_err_list)
def bounded_parameter_err(lb, ub, init_value):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator1_s.bounded_parameter(lb, ub, init_value)


# This test function tests whether set_gp_model throws correct errors
# retrain_count
set_gp_model_err_list = [-1.0, 0.5, "0"]


@pytest.mark.parametrize("retrain_count", set_gp_model_err_list)
def bounded_parameter_err(retrain_count):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator1_s.set_gp_model(retrain_count)


# This test function tests whether get_num_gp_data throws correct errors
# sim_data
get_num_gp_data_err_list = ["sim_data", None, 1]


@pytest.mark.parametrize("sim_data", get_num_gp_data_err_list)
def test_get_num_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail = Type_1_GP_Emulator(
            sim_data,
            val_sse_data2,
            None,
            None,
            None,
            kernel,
            lenscl,
            noise_std,
            outputscl,
            retrain_GP,
            seed,
            normalize,
            None,
            None,
            None,
            None,
        )
        gp_emulator_fail.get_num_gp_data()


# This test function tests whether set_gp_model works correctly
# emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
set_gp_model_list = [
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, 1, 1, 1],
    [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, 1, 1, 1, 1],
    [Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, 1, 1],
    [Type_2_GP_Emulator, sim_data2, val_data2, 1, 1, 1, 1],
    [Type_2_GP_Emulator, sim_data2, val_data2, 2, 1, 2, 1],
    [Type_2_GP_Emulator, sim_data2, val_data2, 2, 2, 2, 2],
]


@pytest.mark.parametrize(
    "gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops",
    set_gp_model_list,
)
def test_set_gp_model(
    gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
):
    gp_emulator = gp_type(
        sim_data,
        val_data,
        None,
        None,
        None,
        kernel,
        lenscl,
        noise_std,
        outputscl,
        retrain_GP,
        seed,
        normalize,
        None,
        None,
        None,
        None,
    )
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert gp_emulator.lenscl == exp_lenscl
    assert gp_emulator.outputscl == exp_ops


# This test function tests whether correct errors get thrown on initialization
# sim_data, val_data, kernel, lenscl, outputscl, retrain_GP
set_gp_model_err_list = [
    [sim_data1, val_data1, "string", 1, 1, 1],
    [sim_data1, val_data1, Kernel_enum(1), 0, 1, 1],
    [sim_data1, val_data1, Kernel_enum(1), 1, 0, 1],
    [sim_data1, val_data1, Kernel_enum(1), 1, 1, -2],
    [sim_data1, val_data1, Kernel_enum(1), -1, 1, 1],
    [sim_data1, val_data1, Kernel_enum(1), 1, -1, 1],
    [sim_data1, val_data1, Kernel_enum(1), 1, 1, -1],
    ["string", val_data1, Kernel_enum(1), 1, 1, 1],
    [sim_data1, "string", Kernel_enum(1), 1, 1, 1],
    [sim_data1, val_data1, Kernel_enum(1), "1", 1, 1],
    [sim_data1, val_data1, Kernel_enum(1), 1, "1", 1],
    [sim_data1, val_data1, Kernel_enum(1), 1, 1, "1"],
    [sim_data1, val_data1, 1, 1, 1, 1],
]


@pytest.mark.parametrize(
    "sim_data, val_data, kernel, lenscl, outputscl, retrain_GP", set_gp_model_err_list
)
def test_set_gp_model_err(sim_data, val_data, kernel, lenscl, outputscl, retrain_GP):
    with pytest.raises((AssertionError, ValueError)):
        gp_emulator = Type_1_GP_Emulator(
            sim_data,
            val_data,
            None,
            None,
            None,
            kernel,
            lenscl,
            noise_std,
            outputscl,
            retrain_GP,
            seed,
            normalize,
            None,
            None,
            None,
            None,
        )


# This test function tests whether get_dim_gp_data checker works correctly
# Emulator class, number of GP training dims
get_dim_gp_data_list = [
    [gp_emulator1_s, 2],
    [gp_emulator1_e, 3],
    [gp_emulator2_s, 4],
    [gp_emulator2_e, 6],
]


@pytest.mark.parametrize("gp_emulator, expected", get_dim_gp_data_list)
def test_get_dim_gp_data(gp_emulator, expected):
    assert gp_emulator.get_dim_gp_data() == expected


# This test function tests whether get_dim_gp_data throws correct errors
# sim_data
get_dim_gp_data_err_list = ["sim_data", None, 1]


@pytest.mark.parametrize("sim_data", get_dim_gp_data_err_list)
def test_get_dim_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail = Type_1_GP_Emulator(
            sim_data,
            val_sse_data2,
            None,
            None,
            None,
            kernel,
            lenscl,
            noise_std,
            outputscl,
            retrain_GP,
            seed,
            normalize,
            None,
            None,
            None,
            None,
        )
        gp_emulator_fail.get_dim_gp_data()


# This test function tests whether set_train_test_data checker works correctly
# gp emulator
set_train_test_data_list = [
    gp_emulator1_s,
    gp_emulator1_e,
    gp_emulator2_s,
    gp_emulator2_e,
]


@pytest.mark.parametrize("gp_emulator", set_train_test_data_list)
def test_set_train_test_data(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    assert len(train_data.theta_vals) + len(test_data.theta_vals) == len(
        gp_emulator.gp_sim_data.theta_vals
    )


# This test function tests whether set_train_test_data throws correct errors
# theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed
set_train_test_data_err_list = [
    [
        None,
        sim_data1.x_vals,
        sim_data1.y_vals,
        sim_data1.bounds_x,
        sim_data1.bounds_theta,
        1,
        1,
    ],
    [
        sim_data1.theta_vals,
        None,
        sim_data1.y_vals,
        sim_data1.bounds_x,
        sim_data1.bounds_theta,
        1,
        1,
    ],
    [
        sim_data1.theta_vals,
        sim_data1.x_vals,
        None,
        sim_data1.bounds_x,
        sim_data1.bounds_theta,
        1,
        1,
    ],
    [
        sim_data1.theta_vals,
        sim_data1.x_vals,
        sim_data1.y_vals,
        None,
        sim_data1.bounds_theta,
        1,
        1,
    ],
    [
        sim_data1.theta_vals,
        sim_data1.x_vals,
        sim_data1.y_vals,
        sim_data1.bounds_x,
        None,
        1,
        1,
    ],
    [
        sim_data1.theta_vals,
        sim_data1.x_vals,
        sim_data1.y_vals,
        sim_data1.bounds_x,
        sim_data1.bounds_theta,
        None,
        1,
    ],
    [
        sim_data1.theta_vals,
        sim_data1.x_vals,
        sim_data1.y_vals,
        sim_data1.bounds_x,
        sim_data1.bounds_theta,
        1,
        None,
    ],
    [None, None, None, None, None, None, None],
    [
        sim_data2.theta_vals,
        sim_data2.x_vals,
        sim_data2.y_vals,
        sim_data2.bounds_x,
        sim_data2.bounds_theta,
        1,
        None,
    ],
]


@pytest.mark.parametrize(
    "theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed",
    set_train_test_data_err_list,
)
def test_get_dim_gp_data_err(
    theta_vals, x_vals, y_vals, bounds_x, bounds_theta, sep_fact, seed
):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        sim_data_fail = Data(
            theta_vals,
            x_vals,
            y_vals,
            None,
            None,
            None,
            None,
            None,
            bounds_theta,
            bounds_x,
            sep_fact,
            seed,
        )

        if all(
            var is None
            for var in [
                theta_vals,
                x_vals,
                y_vals,
                bounds_x,
                bounds_theta,
                sep_fact,
                seed,
            ]
        ):
            sim_data_fail = "string"

        gp_emulator_fail = Type_1_GP_Emulator(
            sim_data_fail,
            val_sse_data2,
            None,
            None,
            None,
            kernel,
            lenscl,
            noise_std,
            outputscl,
            retrain_GP,
            seed,
            normalize,
            None,
            None,
            None,
            None,
        )
        train_data, test_data = gp_emulator_fail.set_train_test_data(
            sim_data_fail.sep_fact, sim_data_fail.seed
        )


# This test function tests whether train_gp checker works correctly
# emulator class type, sim data, val_data, lenscl, outputscl, exp_lenscl, exp_ops
train_gp_list = [
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, 1, np.ones(2), 1],
    [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, 1, 1, np.ones(8), 1],
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, 1, np.ones(2) * 2, 1],
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, 2, np.ones(2) * 2, 2],
]


@pytest.mark.parametrize(
    "gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops", train_gp_list
)
def test_train_gp(gp_type, sim_data, val_data, lenscl, outputscl, exp_lenscl, exp_ops):
    gp_emulator = gp_type(
        sim_data,
        val_data,
        None,
        None,
        None,
        kernel,
        lenscl,
        noise_std,
        outputscl,
        retrain_GP,
        seed,
        normalize,
        None,
        None,
        None,
        None,
    )
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all(np.isclose(gp_emulator.lenscl, exp_lenscl, atol=1e-6, rtol=1e-5))
    assert np.all(np.isclose(trained_ops, exp_ops, atol=1e-6, rtol=1e-5))


# This test function tests whether train_gp checker works correctly (optimizes None parameters between bounds)
# emulator class type, sim data, val_data, lenscl, outputscl, exp_ops
train_gp_opt_list = [
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, None, 1],
    [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, None, 1],
    [Type_1_GP_Emulator, sim_sse_data2, val_sse_data2, None, 2],
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 1, None],
    [Type_1_GP_Emulator, sim_sse_data1, val_sse_data1, 2, None],
]


@pytest.mark.parametrize(
    "gp_type, sim_data, val_data, lenscl, outputscl", train_gp_opt_list
)
def test_train_gp_opt(gp_type, sim_data, val_data, lenscl, outputscl):
    tol = 1e-7
    gp_emulator = gp_type(
        sim_data,
        val_data,
        None,
        None,
        None,
        kernel,
        lenscl,
        noise_std,
        outputscl,
        retrain_GP,
        seed,
        normalize,
        None,
        None,
        None,
        None,
    )
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    trained_lenscl = gp_emulator.trained_hyperparams[0]
    trained_ops = gp_emulator.trained_hyperparams[-1]
    assert gp_emulator.kernel == Kernel_enum.MAT_52
    assert len(trained_lenscl) == gp_emulator.get_dim_gp_data()
    assert np.all(1e-5 - tol <= element <= 1e5 + tol for element in trained_lenscl)
    assert 1e-5 - tol <= trained_ops <= 1e2 + tol


# This test function tests whether train_gp throws correct errors
# gp_emulator, feature_train_data
train_gp_err_list = [[gp_emulator1_s, None], [gp_emulator2_s, None]]


@pytest.mark.parametrize("gp_emulator, feature_train_data", train_gp_err_list)
def test_train_gp_err(gp_emulator, feature_train_data):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
    with pytest.raises((AssertionError, ValueError)):
        if feature_train_data is not True:
            gp_emulator_fail.feature_train_data = feature_train_data

        gp_emulator_fail.train_gp()


# This test function tests whether calc_best_error checker works correctly
# gp emulator
calc_best_error_list = [gp_emulator1_s, gp_emulator2_s]


@pytest.mark.parametrize("gp_emulator", calc_best_error_list)
def test_calc_best_error(gp_emulator):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error, be_theta, train_idc = gp_emulator.calc_best_error()
    assert np.isclose(best_error, min(train_data.y_vals), rtol=1e-6)
    assert np.allclose(
        be_theta, train_data.theta_vals[np.argmin(train_data.y_vals)], rtol=1e-6
    )
    assert train_idc == np.argmin(train_data.y_vals)


# This test function tests whether calc_best_error throws correct errors
# gp_emulator, train_data, y_vals
calc_best_error_err_list = [
    [gp_emulator1_s, None, True],
    [gp_emulator1_s, True, None],
    [gp_emulator2_s, None, True],
    [gp_emulator2_s, True, None],
]


@pytest.mark.parametrize("gp_emulator, train_data, y_vals", calc_best_error_err_list)
def test_calc_best_error_err(gp_emulator, train_data, y_vals):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail = copy.copy(gp_emulator)
        if train_data is True:
            train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
        else:
            gp_emulator_fail.train_data = None

        if y_vals is None:
            gp_emulator_fail.train_data.y_vals = y_vals

        best_error, be_theta, train_idc = gp_emulator_fail.calc_best_error()


# Define exploration bias and set ep_curr
ep_bias = Exploration_Bias(
    ep0, None, Ep_enum(1), None, None, None, None, None, None, None
)
ep_bias.set_ep()

# gp_emulator, exp_data, expected_len
eval_ei_test_list = [[gp_emulator1_s, exp_data1], [gp_emulator2_s, exp_data2]]


@pytest.mark.parametrize("gp_emulator, exp_data", eval_ei_test_list)
def test_eval_ei_test(gp_emulator, exp_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()  # Train model
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test()  # Calc mean, var of gp
    best_error, be_theta, train_idc = gp_emulator.calc_best_error()  # Calc best error
    best_error_metrics = best_error, be_theta, None
    ei = gp_emulator.eval_ei_test(exp_data, ep_bias, best_error_metrics)

    assert len(ei[0]) == len(gp_emulator.test_data.theta_vals)

    # gp_emulator, exp_data, expected_len


eval_ei_val_list = [[gp_emulator1_s, exp_data1], [gp_emulator2_s, exp_data2]]


@pytest.mark.parametrize("gp_emulator, exp_data", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()  # Train model
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val()  # Calc mean, var of gp
    best_error, be_theta, train_idc = gp_emulator.calc_best_error()  # Calc best error
    best_error_metrics = best_error, be_theta, None
    ei = gp_emulator.eval_ei_val(exp_data, ep_bias, best_error_metrics)

    assert len(ei[0]) == len(gp_emulator.gp_val_data.theta_vals)

    # gp_emulator, exp_data, expected_len


eval_ei_cand_list = [
    [gp_emulator1_s, simulator1, exp_data1],
    [gp_emulator2_s, simulator2, exp_data2],
]


@pytest.mark.parametrize("gp_emulator, simulator, exp_data", eval_ei_cand_list)
def test_eval_ei_cand(gp_emulator, simulator, exp_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()  # Train model
    candidate = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    candidate.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set candidate thetas
    gp_emulator.cand_data = candidate  # Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(
        gp_emulator.cand_data
    )  # Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_cand()  # Calc mean, var of gp
    best_error, be_theta, train_idc = gp_emulator.calc_best_error()  # Calc best error
    best_error_metrics = best_error, be_theta, None
    ei = gp_emulator.eval_ei_cand(exp_data, ep_bias, best_error_metrics)

    assert len(ei[0]) == len(gp_emulator.cand_data.theta_vals)

    # gp_emulator, simulator, exp_data


eval_ei_misc_list = [
    [gp_emulator1_s, simulator1, exp_data1],
    [gp_emulator2_s, simulator2, exp_data2],
]


@pytest.mark.parametrize("gp_emulator, simulator, exp_data", eval_ei_misc_list)
def test_eval_ei_misc(gp_emulator, simulator, exp_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()  # Train model
    misc_data = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data)  # Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(
        misc_data, feature_misc_data
    )  # Calc mean, var of gp
    best_error, be_theta, train_idc = gp_emulator.calc_best_error()  # Calc best error
    best_error_metrics = best_error, be_theta, None
    ei = gp_emulator.eval_ei_misc(misc_data, exp_data, ep_bias, best_error_metrics)

    assert len(ei[0]) == len(misc_data.theta_vals)


# This test function tests whether eval_ei_cand/val/test/and misc throw correct errors
# gp_emulator, simualtor, exp_data, ep_bias, best_error, data
calc_best_error_err_list = [
    [gp_emulator1_s, simulator1, exp_data1, ep_bias, 1, None],
    [gp_emulator1_s, simulator1, None, ep_bias, 1, True],
    [gp_emulator1_s, simulator1, exp_data1, None, 1, True],
    [gp_emulator2_s, simulator2, exp_data2, None, 1, True],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, ep_bias, best_error, data",
    calc_best_error_err_list,
)
def test_calc_ei_err(gp_emulator, simulator, exp_data, ep_bias, best_error, data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator_fail = copy.copy(gp_emulator)
    gp_emulator_fail.train_gp()  # Train model

    if data is True and exp_data is not None:
        misc_data = Data(
            None,
            exp_data.x_vals,
            None,
            None,
            None,
            None,
            None,
            None,
            simulator.bounds_theta_reg,
            simulator.bounds_x,
            sep_fact,
            seed,
        )
        misc_data.theta_vals = gp_emulator_fail.gp_sim_data.theta_vals[0].reshape(
            1, -1
        )  # Set misc thetas
        feature_misc_data = gp_emulator_fail.featurize_data(
            misc_data
        )  # Set feature vals
    else:
        data = None

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.cand_data = data  # Set candidate point
        ei = gp_emulator_fail.eval_ei_cand(exp_data, ep_bias, best_error)
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.test_data = data  # Set candidate point
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error)
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.gp_val_data = data  # Set candidate point
        ei = gp_emulator_fail.eval_ei_test(exp_data, ep_bias, best_error)
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        ei = gp_emulator_fail.eval_ei_misc(data, exp_data, ep_bias, best_error)


# Test that featurize_data works as intended
# gp_emulator, simulator, exp_data
featurize_data_list = [
    [gp_emulator1_s, simulator1, exp_data1],
    [gp_emulator2_s, simulator2, exp_data2],
]


@pytest.mark.parametrize("gp_emulator, simulator, exp_data", featurize_data_list)
def test_featurize_data(gp_emulator, simulator, exp_data):
    # Make Fake Data class
    misc_data = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data)  # Set feature vals

    assert gp_emulator.get_dim_gp_data() == feature_misc_data.shape[1]


# Test that featurize_data throws the correct errors
# gp_emulator, simulator, exp_data, bad_data_val
featurize_data_err_list = [
    [gp_emulator1_s, simulator1, exp_data1, None],
    [gp_emulator1_s, simulator1, exp_data1, True],
    [gp_emulator2_s, simulator2, exp_data2, None],
    [gp_emulator2_s, simulator2, exp_data2, True],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, bad_data_val", featurize_data_err_list
)
def test_featurize_data_err(gp_emulator, simulator, exp_data, bad_data_val):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        if bad_data_val is None:
            bad_data = None
        else:
            bounds_theta = simulator.bounds_theta_reg
            bounds_x = simulator.bounds_x
            bad_data = Data(
                None,
                exp_data.x_vals,
                None,
                None,
                None,
                None,
                None,
                None,
                bounds_theta,
                bounds_x,
                sep_fact,
                seed,
            )

        gp_emulator.featurize_data(bad_data)  # Set feature vals


# Test that set_gp_model_data works as intended
# gp_emulator
set_gp_model_data_list = [gp_emulator1_s, gp_emulator2_s]


@pytest.mark.parametrize("gp_emulator", set_gp_model_data_list)
def test_set_gp_model_data(gp_emulator):
    # Make Fake Data class
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    data = gp_emulator.set_gp_model_data()
    assert isinstance(data, tuple)
    assert len(data) == 2
    assert data[0].shape[1] == train_data.theta_vals.shape[1]
    assert len(data[1]) == len(train_data.y_vals)


# Test that set_gp_model_data throws the correct errors
# gp_emulator, simulator, exp_data, bad_data_val
set_gp_model_data_err_list = [gp_emulator1_s, gp_emulator2_s]


@pytest.mark.parametrize("gp_emulator", set_gp_model_data_err_list)
def test_set_gp_model_data_err(gp_emulator):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator.train_data = None
        gp_emulator.feature_train_data = None
        data = gp_emulator.set_gp_model_data()
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
        gp_emulator.train_data.y_vals = None
        data = gp_emulator.set_gp_model_data()


# Define small case study
num_x_data = 5
gen_meth_x = Gen_meth_enum(2)
num_theta_data1 = 20
num_theta_data2 = 20
gen_meth_theta = Gen_meth_enum(1)

ep0 = 1
sep_fact = 0.8
normalize = True
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = None
outputscl = None
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(1))  # 1A

# Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_name1.value, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
sim_sse_data1 = simulator1.sim_data_to_sse_sim_data(
    method, sim_data1, exp_data1, sep_fact
)
val_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
)
val_sse_data1 = simulator1.sim_data_to_sse_sim_data(
    method, val_data1, exp_data1, sep_fact, True
)
gp_emulator1_s = Type_1_GP_Emulator(
    sim_sse_data1,
    val_sse_data1,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)

# Define cs_params, simulator, and exp_data for CS2
simulator2 = simulator_helper_test_fxns(cs_name2.value, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(
    num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
sim_sse_data2 = simulator2.sim_data_to_sse_sim_data(
    method, sim_data2, exp_data2, sep_fact
)
val_data2 = simulator2.gen_sim_data(
    num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
)
val_sse_data2 = simulator2.sim_data_to_sse_sim_data(
    method, val_data2, exp_data2, sep_fact, True
)
gp_emulator2_s = Type_1_GP_Emulator(
    sim_sse_data2,
    val_sse_data2,
    None,
    None,
    None,
    kernel,
    lenscl,
    noise_std,
    outputscl,
    retrain_GP,
    seed,
    normalize,
    None,
    None,
    None,
    None,
)

# #This test function tests whether eval_gp_mean_var checker works correctly
expected_mean1_test = np.array([100.24282692, 135.98039714, 71.01762771, 61.64600428])
expected_var1_test = np.array(
    [10202.05697245, 9917.82147249, 9527.58557615, 8608.78875639]
)
expected_mean2_test = np.array([23.95283024, 30.01293589, 36.52848153, 50.19767961])
expected_var2_test = np.array(
    [2201.65373605, 1958.23848783, 2247.21264886, 2215.19516712]
)
# gp_emulator, covar, expected_mean, expected_var
eval_gp_mean_var_test_list = [
    [gp_emulator1_s, expected_mean1_test, expected_var1_test],
    [gp_emulator2_s, expected_mean2_test, expected_var2_test],
]


@pytest.mark.parametrize(
    "gp_emulator, expected_mean, expected_var", eval_gp_mean_var_test_list
)
def test_eval_gp_mean_var_test(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_mean, gp_covar = gp_emulator.eval_gp_mean_var_test(True)  # Calc mean, var of gp
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test(False)  # Calc mean, var of gp

    assert len(gp_mean) == len(test_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    assert len(gp_covar.shape) == 2
    assert gp_covar.shape[0] == gp_covar.shape[1]


expected_mean1_val = np.array(
    [
        137.16501802,
        26.28334912,
        129.29809631,
        60.28970041,
        24.82357534,
        61.64600428,
        166.50576753,
        48.9688545,
        100.24282692,
        38.59820844,
        80.67352095,
        71.01762771,
        135.98039714,
        37.18172925,
        160.13240386,
        68.44368666,
        188.25161374,
        22.4132505,
        113.1800697,
        69.44015394,
    ]
)

expected_var1_val = np.array(
    [
        9167.40640821,
        8997.00050032,
        8154.01907319,
        8837.80051459,
        8589.04914073,
        8608.78875639,
        8490.52341277,
        8268.09743853,
        10202.05697245,
        8469.79644674,
        8337.11229867,
        9527.58557615,
        9917.82147249,
        8946.32376448,
        9149.84124185,
        8392.70472609,
        9274.57443785,
        8331.33119991,
        8506.89239597,
        9327.57554819,
    ]
)

expected_mean2_val = np.array(
    [
        32.80688712,
        19.58235213,
        19.33791041,
        201.07007708,
        52.13411713,
        50.19767961,
        43.48283458,
        57.1623352,
        23.95283024,
        16.16785103,
        32.10258467,
        36.52848153,
        30.01293589,
        45.01141881,
        17.06523779,
        29.32228386,
        17.20234858,
        96.20342979,
        125.00434596,
        17.89258104,
    ]
)

expected_var2_val = np.array(
    [
        1698.94874584,
        1714.60320706,
        1666.03346941,
        1716.36756758,
        1644.17772403,
        2215.19516712,
        1676.30624446,
        1691.76156054,
        2201.65373605,
        1708.27175743,
        1709.60311048,
        2247.21264886,
        1958.23848783,
        1702.35957295,
        1693.54053649,
        1679.76326966,
        1699.15377815,
        1676.43093479,
        1658.74128014,
        1728.38112734,
    ]
)

# gp_emulator, expected_mean, expected_var
eval_gp_mean_var_val_list = [
    [gp_emulator1_s, expected_mean1_val, expected_var1_val],
    [gp_emulator2_s, expected_mean2_val, expected_var2_val],
]


@pytest.mark.parametrize(
    "gp_emulator, expected_mean, expected_var", eval_gp_mean_var_val_list
)
def test_eval_gp_mean_var_val(gp_emulator, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val(False)  # Calc mean, var of gp
    gp_mean, gp_covar = gp_emulator.eval_gp_mean_var_val(True)  # Calc mean, var of gp

    assert len(gp_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    assert len(gp_covar.shape) == 2
    assert gp_covar.shape[0] == gp_covar.shape[1]


expected_mean1 = np.array([137.16501802])
expected_var1 = np.array([9167.40640821])
expected_mean2 = np.array([32.80688712])
expected_var2 = np.array([1698.94874584])

# gp_emulator, covar, simulator, exp_data, expected_mean, expected_var
eval_gp_mean_var_misc_list = [
    [gp_emulator1_s, False, simulator1, exp_data1, expected_mean1, expected_var1],
    [gp_emulator1_s, True, simulator1, exp_data1, expected_mean1, expected_var1],
    [gp_emulator2_s, False, simulator2, exp_data2, expected_mean2, expected_var2],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, simulator, exp_data, expected_mean, expected_var",
    eval_gp_mean_var_misc_list,
)
def test_eval_gp_mean_var_misc_cand(
    gp_emulator, covar, simulator, exp_data, expected_mean, expected_var
):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    misc_data = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data)  # Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(
        misc_data, feature_misc_data, covar
    )  # Calc mean, var of gp

    candidate = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    candidate.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set candidate thetas
    gp_emulator.cand_data = candidate  # Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(
        gp_emulator.cand_data
    )  # Set feature vals
    gp_mean_cand, gp_var_cand = gp_emulator.eval_gp_mean_var_cand(
        covar
    )  # Calc mean, var of gp

    assert (
        len(gp_mean)
        == len(misc_data.theta_vals)
        == len(gp_var)
        == len(gp_mean_cand)
        == len(gp_var_cand)
    )
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)
    assert np.allclose(gp_mean_cand, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(gp_var, expected_var, rtol=1e-02)
        assert np.allclose(gp_var_cand, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    else:
        assert len(gp_var.shape) == 2
        assert len(gp_var_cand.shape) == 2
        assert gp_var.shape[0] == gp_var.shape[1]
        assert gp_var_cand.shape[0] == gp_var_cand.shape[1]


# This function tests whether eval_gp_mean_var_test/val/cand/misc throw the correct errors
# gp_emulator
eval_gp_mean_var_err_list = [gp_emulator1_s, gp_emulator2_s]


@pytest.mark.parametrize("gp_emulator", eval_gp_mean_var_err_list)
def test_eval_gp_mean_var_err(gp_emulator):
    gp_emulator_fail = copy.copy(gp_emulator)
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.feature_val_data = None
        gp_mean, gp_var = (
            gp_emulator_fail.eval_gp_mean_var_val()
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.feature_test_data = None
        gp_mean, gp_var = (
            gp_emulator_fail.eval_gp_mean_var_test()
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.feature_cand_data = None
        gp_mean, gp_var = (
            gp_emulator_fail.eval_gp_mean_var_cand()
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        misc_data = None
        feat_misc_data = None
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(
            misc_data, feat_misc_data
        )  # Calc mean, var of gp

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_val(
            "False"
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test(
            0
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand(
            "True"
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(
            misc_data, feat_misc_data, 1
        )  # Calc mean, var of gp


# This test function tests whether eval_gp_sse_var checker works correctly
# Define exploration bias and set ep_curr
# gp_emulator, covar, expected_mean, expected_var
eval_gp_sse_var_test_list = [
    [gp_emulator1_s, False, expected_mean1_test, expected_var1_test],
    [gp_emulator1_s, True, expected_mean1_test, expected_var1_test],
    [gp_emulator2_s, False, expected_mean2_test, expected_var2_test],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, expected_mean, expected_var", eval_gp_sse_var_test_list
)
def test_eval_gp_sse_var_test(gp_emulator, covar, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test()  # Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test(covar)

    assert len(sse_mean) == len(test_data.theta_vals) == len(sse_var)
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2
        assert sse_var.shape[0] == sse_var.shape[1]

        # gp_emulator, covar, expected_mean, expected_var


eval_gp_sse_var_val_list = [
    [gp_emulator1_s, False, expected_mean1_val, expected_var1_val],
    [gp_emulator1_s, True, expected_mean1_val, expected_var1_val],
    [gp_emulator2_s, False, expected_mean2_val, expected_var2_val],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, expected_mean, expected_var", eval_gp_sse_var_val_list
)
def test_eval_gp_sse_var_val(gp_emulator, covar, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val()  # Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val(covar)

    assert len(sse_mean) == len(gp_emulator.gp_val_data.theta_vals) == len(sse_var)
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2
        assert sse_var.shape[0] == sse_var.shape[1]

        # gp_emulator, covar, simulator, exp_data, expected_mean, expected_var


eval_gp_sse_var_misc_list = [
    [gp_emulator1_s, False, simulator1, exp_data1, expected_mean1, expected_var1],
    [gp_emulator1_s, True, simulator1, exp_data1, expected_mean1, expected_var1],
    [gp_emulator2_s, False, simulator2, exp_data2, expected_mean2, expected_var2],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, simulator, exp_data, expected_mean, expected_var",
    eval_gp_sse_var_misc_list,
)
def test_eval_gp_sse_var_misc_cand(
    gp_emulator, covar, simulator, exp_data, expected_mean, expected_var
):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    misc_data = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    misc_data.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data)  # Set feature vals
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(
        misc_data, feature_misc_data
    )  # Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_misc(misc_data, covar)

    candidate = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    candidate.theta_vals = gp_emulator.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set candidate thetas
    gp_emulator.cand_data = candidate  # Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(
        gp_emulator.cand_data
    )  # Set feature vals
    gp_mean_cand, gp_var_cand = (
        gp_emulator.eval_gp_mean_var_cand()
    )  # Calc mean, var of gp
    sse_mean_cand, sse_var_cand = gp_emulator.eval_gp_sse_var_cand(covar)

    assert (
        len(sse_mean)
        == len(misc_data.theta_vals)
        == len(sse_var)
        == len(sse_mean_cand)
        == len(sse_var_cand)
    )
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
        assert np.allclose(sse_var_cand, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2
        assert len(sse_var_cand.shape) == 2
        assert sse_var.shape[0] == sse_var.shape[1]
        assert sse_var_cand.shape[0] == sse_var_cand.shape[1]


# This function tests whether eval_gp_sse_var_test/val/cand/misc throw the correct errors
# gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, set_covar
eval_gp_sse_var_err_list = [
    [gp_emulator1_s, simulator1, exp_data1, False, True, True, True],
    [gp_emulator1_s, simulator1, exp_data1, True, False, True, True],
    [gp_emulator1_s, simulator1, exp_data1, True, True, False, True],
    [gp_emulator1_s, simulator1, exp_data1, True, True, True, False],
    [gp_emulator2_s, simulator2, exp_data2, True, True, False, True],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, set_covar",
    eval_gp_sse_var_err_list,
)
def test_eval_gp_sse_var_err(
    gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, set_covar
):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
    gp_emulator_fail.train_gp()

    candidate = Data(
        None,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    candidate.theta_vals = gp_emulator_fail.gp_sim_data.theta_vals[0].reshape(
        1, -1
    )  # Set candidate thetas
    gp_emulator_fail.cand_data = candidate  # Set candidate point
    gp_emulator_fail.feature_cand_data = gp_emulator_fail.featurize_data(
        gp_emulator_fail.cand_data
    )  # Set feature vals

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_val()
        if set_data is False:
            gp_emulator_fail.gp_val_data = None
        if set_gp_mean is False:
            gp_emulator_fail.gp_val_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.gp_val_data.gp_var = None
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val("False")
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val()
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test()
        if set_data is False:
            gp_emulator_fail.test_data = None
        if set_gp_mean is False:
            gp_emulator_fail.test_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.test_data.gp_var = None

        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test("False")
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test()

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand()
        if set_data is False:
            gp_emulator_fail.cand_data = None
        if set_gp_mean is False:
            gp_emulator_fail.cand_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.cand_data.gp_var = None

        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand("False")
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand()

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(
            misc_data, feat_misc_data
        )  # Calc mean, var of gp
        if set_data is False:
            misc_data = None
        if set_gp_mean is False:
            misc_data.gp_mean = None
        if set_gp_var is False:
            misc_data.gp_var = None

        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(
                misc_data, "False"
            )
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(misc_data)


# Test that add_next_theta_to_train_data(theta_best_sse_data) works correctly
# gp_emulator, simulator, exp_data, expected_mean, expected_var
add_next_theta_to_train_data_list = [
    [gp_emulator1_s, simulator1, exp_data1],
    [gp_emulator2_s, simulator2, exp_data2],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data", add_next_theta_to_train_data_list
)
def test_add_next_theta_to_train_data(gp_emulator, simulator, exp_data):
    # Get number of training data before
    theta_before = len(gp_emulator.train_data.theta_vals)
    # Create fake theta_best_sse_data
    theta_best = gp_emulator.gp_sim_data.theta_vals[0]
    theta_best_repeated = np.repeat(
        theta_best.reshape(1, -1), exp_data.get_num_x_vals(), axis=0
    )
    # Add instance of Data class to theta_best
    theta_best_data = Data(
        theta_best_repeated,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    # Calculate y values and sse for theta_best with noise
    theta_best_data.y_vals = simulator.gen_y_data(
        theta_best_data, simulator.noise_mean, simulator.noise_std
    )
    # Set the best data to be in sse form if using a type 1 GP
    theta_best_data = simulator.sim_data_to_sse_sim_data(
        method, theta_best_data, exp_data, sep_fact, False
    )

    # Append training data
    gp_emulator.add_next_theta_to_train_data(theta_best_data)

    assert len(gp_emulator.train_data.theta_vals) == theta_before + 1


# Test that add_next_theta_to_train_data(theta_best_sse_data) throws correct errors correctly
train_data1 = Data(
    sim_data1.theta_vals,
    sim_data1.x_vals,
    None,
    None,
    None,
    None,
    None,
    None,
    simulator1.bounds_theta_reg,
    simulator1.bounds_x,
    sep_fact,
    seed,
)
train_data2 = Data(
    None,
    sim_data1.x_vals,
    sim_data1.y_vals,
    None,
    None,
    None,
    None,
    None,
    simulator1.bounds_theta_reg,
    simulator1.bounds_x,
    sep_fact,
    seed,
)
train_data3 = "str"
theta_best_data1 = Data(
    sim_data1.theta_vals,
    sim_data1.x_vals,
    None,
    None,
    None,
    None,
    None,
    None,
    simulator1.bounds_theta_reg,
    simulator1.bounds_x,
    sep_fact,
    seed,
)
theta_best_data2 = Data(
    None,
    sim_data1.x_vals,
    sim_data1.y_vals,
    None,
    None,
    None,
    None,
    None,
    simulator1.bounds_theta_reg,
    simulator1.bounds_x,
    sep_fact,
    seed,
)
theta_best_data3 = "str"


# Test that add_next_theta_to_train_data(theta_best_sse_data) works correctly
# gp_emulator, simulator, exp_data, bad_new_data, bad_train_data
add_next_theta_to_train_data_list = [
    [gp_emulator1_s, simulator1, exp_data1, None, train_data1],
    [gp_emulator1_s, simulator1, exp_data1, None, train_data2],
    [gp_emulator1_s, simulator1, exp_data1, None, train_data3],
    [gp_emulator1_s, simulator1, exp_data1, theta_best_data1, None],
    [gp_emulator1_s, simulator1, exp_data1, theta_best_data2, None],
    [gp_emulator1_s, simulator1, exp_data1, theta_best_data3, None],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, bad_new_data, bad_train_data",
    add_next_theta_to_train_data_list,
)
def test_add_next_theta_to_train_data(
    gp_emulator, simulator, exp_data, bad_new_data, bad_train_data
):
    gp_emulator_fail = copy.copy(gp_emulator)
    # Create fake theta_best_sse_data
    theta_best = gp_emulator_fail.gp_sim_data.theta_vals[0]
    theta_best_repeated = np.repeat(
        theta_best.reshape(1, -1), exp_data.get_num_x_vals(), axis=0
    )
    # Add instance of Data class to theta_best
    theta_best_data = Data(
        theta_best_repeated,
        exp_data.x_vals,
        None,
        None,
        None,
        None,
        None,
        None,
        simulator.bounds_theta_reg,
        simulator.bounds_x,
        sep_fact,
        seed,
    )
    # Calculate y values and sse for theta_best with noise
    theta_best_data.y_vals = simulator.gen_y_data(
        theta_best_data, simulator.noise_mean, simulator.noise_std
    )
    # Set the best data to be in sse form if using a type 1 GP
    theta_best_data = simulator.sim_data_to_sse_sim_data(
        method, theta_best_data, exp_data, sep_fact, False
    )

    if bad_new_data is not None:
        theta_best_data = bad_new_data

    if bad_train_data is not None:
        gp_emulator_fail.train_data = bad_train_data

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        # Append training data
        gp_emulator_fail.add_next_theta_to_train_data(theta_best_data)
