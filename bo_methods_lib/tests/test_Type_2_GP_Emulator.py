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
cs_val1 = 1
cs_val2 = 2

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
noise_std = 0
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(3))  # 2A

# Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_val1, noise_mean, noise_std, seed)
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
simulator2 = simulator_helper_test_fxns(cs_val2, noise_mean, noise_std, seed)
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
get_num_gp_data_list = [[gp_emulator1_e, 125], [gp_emulator2_e, 400]]


@pytest.mark.parametrize("gp_emulator, expected", get_num_gp_data_list)
def test_get_num_gp_data(gp_emulator, expected):
    assert gp_emulator.get_num_gp_data() == expected


# This test function tests whether get_num_gp_data throws correct errors
# sim_data
get_num_gp_data_err_list = ["sim_data", None, 1]


@pytest.mark.parametrize("sim_data", get_num_gp_data_err_list)
def test_get_num_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail = Type_2_GP_Emulator(
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
]


@pytest.mark.parametrize(
    "sim_data, val_data, kernel, lenscl, outputscl, retrain_GP", set_gp_model_err_list
)
def test_set_gp_model_err(sim_data, val_data, kernel, lenscl, outputscl, retrain_GP):
    with pytest.raises((AssertionError, ValueError)):
        gp_emulator = Type_2_GP_Emulator(
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
get_dim_gp_data_list = [[gp_emulator1_e, 3], [gp_emulator2_e, 6]]


@pytest.mark.parametrize("gp_emulator, expected", get_dim_gp_data_list)
def test_get_dim_gp_data(gp_emulator, expected):
    assert gp_emulator.get_dim_gp_data() == expected


# This test function tests whether get_dim_gp_data throws correct errors
# sim_data
get_dim_gp_data_err_list = ["sim_data", None, 1]


@pytest.mark.parametrize("sim_data", get_dim_gp_data_err_list)
def test_get_dim_gp_data_err(sim_data):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail = Type_2_GP_Emulator(
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
# gp emulator, cs_params
set_train_test_data_list = [gp_emulator1_e, gp_emulator2_e]


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

        gp_emulator_fail = Type_2_GP_Emulator(
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
# For time sake does not consider cs2 when testing
train_gp_list = [
    [Type_2_GP_Emulator, sim_data1, val_data1, 1, 1, np.ones(3), 1],
    [Type_2_GP_Emulator, sim_data1, val_data1, 2, 1, np.ones(3) * 2, 1],
    [Type_2_GP_Emulator, sim_data1, val_data1, 2, 2, np.ones(3) * 2, 2],
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
    [Type_2_GP_Emulator, sim_data1, val_data1, None, 1],
    [Type_2_GP_Emulator, sim_data1, val_data1, 1, None],
    [Type_2_GP_Emulator, sim_data1, val_data1, 2, None],
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


# This test function tests whether train_gp throws correct errors
# gp_emulator, feature_train_data
train_gp_err_list = [[gp_emulator1_e, None], [gp_emulator2_e, None]]


@pytest.mark.parametrize("gp_emulator, feature_train_data", train_gp_err_list)
def test_train_gp_err(gp_emulator, feature_train_data):
    gp_emulator_fail = copy.copy(gp_emulator)
    train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
    with pytest.raises((AssertionError, ValueError)):
        if feature_train_data is not True:
            gp_emulator_fail.feature_train_data = feature_train_data

        gp_emulator_fail.train_gp()


# This test function tests whether calc_best_error checker works correctly
# gp emulator, exp_data, sim_sse_data
calc_best_error_list = [
    [gp_emulator1_e, exp_data1, sim_sse_data1],
    [gp_emulator2_e, exp_data2, sim_sse_data2],
]


@pytest.mark.parametrize("gp_emulator, exp_data, sim_sse_data", calc_best_error_list)
def test_calc_best_error(gp_emulator, exp_data, sim_sse_data):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(
        method, exp_data
    )
    assert np.isclose(best_error, float(min(sim_sse_data.y_vals)), rtol=1e-6)
    assert np.allclose(
        be_theta, sim_sse_data.theta_vals[np.argmin(sim_sse_data.y_vals)], rtol=1e-6
    )
    assert np.isclose(sum(best_sq_error), float(min(sim_sse_data.y_vals)), rtol=1e-6)


# This test function tests whether calc_best_error throws correct errors
# gp_emulator, exp_data, train_data, y_vals, method
calc_best_error_err_list = [
    [gp_emulator1_e, exp_data1, None, True, method],
    [gp_emulator1_e, exp_data1, True, None, method],
    [gp_emulator1_e, None, True, True, method],
    [gp_emulator2_e, exp_data2, None, True, method],
    [gp_emulator2_e, exp_data2, True, None, method],
    [gp_emulator2_e, exp_data2, True, True, None],
]


@pytest.mark.parametrize(
    "gp_emulator, exp_data, train_data, y_vals, method", calc_best_error_err_list
)
def test_calc_best_error_err(gp_emulator, exp_data, train_data, y_vals, method):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail = copy.copy(gp_emulator)
        if train_data is True:
            train_data, test_data = gp_emulator_fail.set_train_test_data(sep_fact, seed)
        else:
            gp_emulator_fail.train_data = None

        if y_vals is None:
            gp_emulator_fail.train_data.y_vals = y_vals

        best_error = gp_emulator_fail.calc_best_error(method, exp_data)


# This test function tests whether eval_gp_ei works correctly
# Define exploration bias and set ep_curr
ep_bias = Exploration_Bias(
    ep0, None, Ep_enum(1), None, None, None, None, None, None, None
)
ep_bias.set_ep()

# gp_emulator, exp_data, method
eval_ei_test_list = [
    [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(3))],
    [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(4))],
]


@pytest.mark.parametrize("gp_emulator, exp_data, method", eval_ei_test_list)
def test_eval_ei_test(gp_emulator, exp_data, method):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()  # Train model
    # Set testing data to training data for example
    gp_emulator.feature_test_data = gp_emulator.featurize_data(gp_emulator.train_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test()  # Calc mean, var of gp
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(
        method, exp_data
    )  # Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    sg_mc_samples = 2000
    ei = gp_emulator.eval_ei_test(
        exp_data, ep_bias, best_error_metrics, method, sg_mc_samples
    )
    # Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0]) * num_x_data == len(gp_emulator.train_data.theta_vals)

    # gp_emulator, exp_data, method


eval_ei_val_list = [
    [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(3))],
    [gp_emulator1_e, exp_data1, GPBO_Methods(Method_name_enum(4))],
]


@pytest.mark.parametrize("gp_emulator, exp_data, method_val", eval_ei_val_list)
def test_eval_ei_val(gp_emulator, exp_data, method_val):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()  # Train model
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_val()  # Calc mean, var of gp
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(
        method_val, exp_data
    )  # Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    ei = gp_emulator.eval_ei_val(
        exp_data, ep_bias, best_error_metrics, method_val, sg_mc_samples=2000
    )
    # Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0]) * num_x_data == len(gp_emulator.gp_val_data.theta_vals)

    # gp_emulator, exp_data, method


eval_ei_cand_list = [
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))],
]


@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", eval_ei_cand_list)
def test_eval_ei_cand(gp_emulator, simulator, exp_data, method):
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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate  # Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(gp_emulator.cand_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_cand()  # Calc mean, var of gp
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(
        method, exp_data
    )  # Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    ei = gp_emulator.eval_ei_cand(
        exp_data, ep_bias, best_error_metrics, method, sg_mc_samples=2000
    )
    # Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0]) * num_x_data == len(gp_emulator.cand_data.theta_vals)

    # gp_emulator, exp_data, method


eval_ei_misc_list = [
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))],
]


@pytest.mark.parametrize("gp_emulator, simulator, exp_data, method", eval_ei_misc_list)
def test_eval_ei_misc(gp_emulator, simulator, exp_data, method):
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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "misc_data thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    misc_data.theta_vals = theta_vals
    feature_misc_data = gp_emulator.featurize_data(misc_data)
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_misc(
        misc_data, feature_misc_data
    )  # Calc mean, var of gp
    best_error, be_theta, best_sq_error, org_train_idcs = gp_emulator.calc_best_error(
        method, exp_data
    )  # Calc best error
    best_error_metrics = best_error, be_theta, best_sq_error
    ei = gp_emulator.eval_ei_misc(
        misc_data, exp_data, ep_bias, best_error_metrics, method, sg_mc_samples=2000
    )
    # Multiply by 5 because there is 1 prediction for each x data point
    assert len(ei[0]) * num_x_data == len(misc_data.theta_vals)


# This test function tests whether eval_ei_cand/val/test/and misc throw correct errors
# gp_emulator, simualtor, exp_data, ep_bias, method, data
calc_ei_err_list = [
    [gp_emulator1_e, simulator1, exp_data1, ep_bias, method, None],
    [gp_emulator1_e, simulator1, None, ep_bias, method, True],
    [
        gp_emulator1_e,
        simulator1,
        exp_data1,
        ep_bias,
        GPBO_Methods(Method_name_enum(5)),
        True,
    ],
    [
        gp_emulator1_e,
        simulator1,
        exp_data1,
        ep_bias,
        GPBO_Methods(Method_name_enum(6)),
        True,
    ],
    [
        gp_emulator1_e,
        simulator1,
        exp_data1,
        ep_bias,
        GPBO_Methods(Method_name_enum(1)),
        True,
    ],
    [gp_emulator1_e, simulator1, exp_data1, ep_bias, "str", True],
    [gp_emulator1_e, simulator1, exp_data1, None, method, True],
    [gp_emulator2_e, simulator2, exp_data2, None, method, True],
    [gp_emulator2_e, simulator2, exp_data2, ep_bias, None, True],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, ep_bias, method, data", calc_ei_err_list
)
def test_calc_ei_err(gp_emulator, simulator, exp_data, ep_bias, method, data):
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
        theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
            1, -1
        )  # Set "misc_data thetas"
        theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
        misc_data.theta_vals = theta_vals
        feature_misc_data = gp_emulator.featurize_data(misc_data)
        if isinstance(method, GPBO_Methods):
            best_error, be_theta, best_sq_error, org_train_idcs = (
                gp_emulator.calc_best_error(method, exp_data)
            )  # Calc best error
            best_error_metrics = best_error, be_theta, best_sq_error
        else:
            best_error_metrics = None
    else:
        data = None
        best_error_metrics = None

    if method not in [
        GPBO_Methods(Method_name_enum(5)),
        GPBO_Methods(Method_name_enum(6)),
    ]:
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.cand_data = data  # Set candidate point
            ei = gp_emulator_fail.eval_ei_cand(
                exp_data, ep_bias, best_error_metrics, method
            )
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.gp_val_data = data  # Set candidate point
            ei = gp_emulator_fail.eval_ei_val(
                exp_data, ep_bias, best_error_metrics, method
            )
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator_fail.test_data = data  # Set candidate point
        ei = gp_emulator_fail.eval_ei_test(
            exp_data, ep_bias, best_error_metrics, method
        )
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        ei = gp_emulator_fail.eval_ei_misc(
            data, exp_data, ep_bias, best_error_metrics, method
        )

    if method in [GPBO_Methods(Method_name_enum(5)), GPBO_Methods(Method_name_enum(6))]:
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.cand_data = data  # Set candidate point
            ei = gp_emulator_fail.eval_ei_cand(
                exp_data, ep_bias, best_error_metrics, method, 0
            )
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.test_data = data  # Set candidate point
            ei = gp_emulator_fail.eval_ei_test(
                exp_data, ep_bias, best_error_metrics, method, 0.8
            )
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            gp_emulator_fail.gp_val_data = data  # Set candidate point
            ei = gp_emulator_fail.eval_ei_val(
                exp_data, ep_bias, best_error_metrics, method, "1"
            )
        with pytest.raises((AssertionError, AttributeError, ValueError)):
            ei = gp_emulator_fail.eval_ei_misc(
                data, exp_data, ep_bias, best_error_metrics, method, None
            )

            # gp_emulator, exp_data, method


featurize_data_list = [
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(3))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(4))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(5))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(6))],
    [gp_emulator1_e, simulator1, exp_data1, GPBO_Methods(Method_name_enum(7))],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, method", featurize_data_list
)
def test_featurize_data(gp_emulator, simulator, exp_data, method):
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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "misc_data thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    misc_data.theta_vals = theta_vals
    feature_misc_data = gp_emulator.featurize_data(misc_data)

    # Multiply by 5 because there is 1 prediction for each x data point
    assert gp_emulator.get_dim_gp_data() == feature_misc_data.shape[1]


# Test that featurize_data throws the correct errors
# gp_emulator, simulator, exp_data, bad_data_val
featurize_data_err_list = [
    [gp_emulator1_e, simulator1, exp_data1, None],
    [gp_emulator1_e, simulator1, exp_data1, True],
    [gp_emulator2_e, simulator2, exp_data2, None],
    [gp_emulator2_e, simulator2, exp_data2, True],
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
set_gp_model_data_list = [gp_emulator1_e, gp_emulator2_e]


@pytest.mark.parametrize("gp_emulator", set_gp_model_data_list)
def test_set_gp_model_data(gp_emulator):
    # Make Fake Data class
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    data = gp_emulator.set_gp_model_data()
    assert isinstance(data, tuple)
    assert len(data) == 2
    assert (
        data[0].shape[1] == train_data.theta_vals.shape[1] + train_data.x_vals.shape[1]
    )
    assert len(data[1]) == len(train_data.y_vals)


# Test that set_gp_model_data throws the correct errors
# gp_emulator, simulator, exp_data, bad_data_val
set_gp_model_data_err_list = [gp_emulator1_e, gp_emulator2_e]


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

# This test function tests whether set_gp_model throws correct errors
# retrain_count
set_gp_model_err_list = [-1.0, 0.5, "0"]


@pytest.mark.parametrize("retrain_count", set_gp_model_err_list)
def bounded_parameter_err(retrain_count):
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_emulator1_e.set_gp_model(retrain_count)

# Define small case study
num_x_data = 5
gen_meth_x = Gen_meth_enum(1)
num_theta_data1 = 10
num_theta_data2 = 5
gen_meth_theta = Gen_meth_enum(1)

ep0 = 1
sep_fact = 0.8
normalize = True
noise_mean = 0
noise_std = 0.01
kernel = Kernel_enum(1)
lenscl = 1
outputscl = 1
retrain_GP = 0
seed = 1
method = GPBO_Methods(Method_name_enum(3))  # 2A

# Define cs_params, simulator, and exp_data for CS1
simulator1 = simulator_helper_test_fxns(cs_val2, noise_mean, noise_std, seed)
exp_data1 = simulator1.gen_exp_data(num_x_data, gen_meth_x)
sim_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
val_data1 = simulator1.gen_sim_data(
    num_theta_data1, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
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
simulator2 = simulator_helper_test_fxns(cs_val2, noise_mean, noise_std, seed)
exp_data2 = simulator2.gen_exp_data(num_x_data, gen_meth_x)
sim_data2 = simulator2.gen_sim_data(
    num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact
)
val_data2 = simulator2.gen_sim_data(
    num_theta_data2, num_x_data, gen_meth_theta, gen_meth_x, sep_fact, True
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

# This test function tests whether eval_gp_mean_var checker works correctly
expected_mean1_test = np.array(
    [
        5.34660597,
        5.34258815,
        5.39685485,
        5.33272659,
        5.34497003,
        5.40847268,
        5.53064388,
        5.32786468,
        5.32655377,
        5.53983541,
    ]
)
expected_var1_test = np.array(
    [
        0.44038916,
        0.42503268,
        0.43802073,
        0.4383011,
        0.42626707,
        0.48903631,
        0.48665984,
        0.48848317,
        0.48860143,
        0.48703375,
    ]
)

expected_mean2_test = np.array(
    [5.36303006, 5.37938932, 5.42775988, 5.42508199, 5.37980021]
)

expected_var2_test = np.array(
    [0.29768982, 0.29753789, 0.29764514, 0.29766055, 0.29757462]
)
# gp_emulator, expected_mean, expected_var
eval_gp_mean_var_test_list = [
    [gp_emulator1_e, False, expected_mean1_test, expected_var1_test],
    [gp_emulator1_e, True, expected_mean1_test, expected_var1_test],
    [gp_emulator2_e, False, expected_mean2_test, expected_var2_test],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, expected_mean, expected_var", eval_gp_mean_var_test_list
)
def test_eval_gp_mean_var_test(gp_emulator, covar, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_mean, gp_var = gp_emulator.eval_gp_mean_var_test(False)  # Calc mean, var of gp
    gp_mean, gp_covar = gp_emulator.eval_gp_mean_var_test(True)  # Calc mean, var of gp

    assert len(gp_mean) == len(test_data.theta_vals) == len(gp_var)
    assert np.allclose(gp_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    assert np.allclose(gp_var, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    assert len(gp_covar.shape) == 2
    assert gp_covar.shape[0] == gp_covar.shape[1]


expected_mean1_val = np.array(
    [
        5.43339178,
        5.93714794,
        5.10506322,
        5.07267849,
        5.99487708,
        5.67105833,
        5.45988983,
        7.44999756,
        6.86453322,
        5.38567659,
        5.36145721,
        5.34759801,
        5.48975736,
        5.37220496,
        5.34599559,
        6.21906882,
        5.58401529,
        8.27572816,
        7.58450352,
        5.44988767,
        5.35958496,
        5.32977543,
        5.64179967,
        5.56339363,
        5.32402512,
        5.34660597,
        5.34258815,
        5.39685485,
        5.33272659,
        5.34497003,
        5.35442828,
        5.22748775,
        5.21849787,
        4.96899975,
        5.25923496,
        5.44400095,
        5.17789512,
        6.2040906,
        5.85048901,
        5.1793587,
        5.40847268,
        5.53064388,
        5.32786468,
        5.32655377,
        5.53983541,
        5.4726564,
        5.39275907,
        7.24713287,
        6.65999134,
        5.33146982,
    ]
)

expected_var1_val = np.array(
    [
        0.3680345,
        0.33698742,
        0.36472418,
        0.3650008,
        0.33846738,
        0.36573115,
        0.33551052,
        0.36256839,
        0.36282951,
        0.33694734,
        0.37060778,
        0.33863062,
        0.36712618,
        0.36742155,
        0.34015609,
        0.37013459,
        0.33833489,
        0.36668731,
        0.36697886,
        0.33985218,
        0.37068652,
        0.3386801,
        0.36719948,
        0.36749514,
        0.3402064,
        0.44038916,
        0.42503268,
        0.43802073,
        0.4383011,
        0.42626707,
        0.36749941,
        0.33665424,
        0.36422772,
        0.36450079,
        0.33812521,
        0.36755201,
        0.33674599,
        0.36429578,
        0.36456577,
        0.33820959,
        0.48903631,
        0.48665984,
        0.48848317,
        0.48860143,
        0.48703375,
        0.36294603,
        0.33369474,
        0.35995617,
        0.36020017,
        0.33508382,
    ]
)

expected_mean2_val = np.array(
    [
        5.56656896,
        6.59117289,
        4.91921459,
        5.3213033,
        6.68196096,
        6.02451175,
        5.36223488,
        7.84506873,
        7.23750481,
        5.21103654,
        4.93263083,
        5.27864953,
        5.47341357,
        5.46455557,
        5.31600303,
        5.36303006,
        5.37938932,
        5.42775988,
        5.42508199,
        5.37980021,
        5.32498157,
        5.39634398,
        5.24900595,
        5.35555801,
        5.41699245,
    ]
)

expected_var2_val = np.array(
    [
        0.22245441,
        0.20320591,
        0.22035304,
        0.22053106,
        0.20412299,
        0.22246238,
        0.20321105,
        0.22036052,
        0.22053854,
        0.20412815,
        0.22223726,
        0.2030682,
        0.22015056,
        0.2203274,
        0.20398264,
        0.29768982,
        0.29753789,
        0.29764514,
        0.29766055,
        0.29757462,
        0.22221959,
        0.20305698,
        0.220134,
        0.22031089,
        0.20397138,
    ]
)

# gp_emulator, covar expected_mean, expected_var
eval_gp_mean_var_val_list = [
    [gp_emulator1_e, False, expected_mean1_val, expected_var1_val],
    [gp_emulator1_e, True, expected_mean1_val, expected_var1_val],
    [gp_emulator2_e, False, expected_mean2_val, expected_var2_val],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, expected_mean, expected_var", eval_gp_mean_var_val_list
)
def test_eval_gp_mean_var_val(gp_emulator, covar, expected_mean, expected_var):
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


expected_mean1 = np.array([5.43339178, 5.93714794, 5.10506322, 5.07267849, 5.99487708])
expected_var1 = np.array([0.3680345, 0.33698742, 0.36472418, 0.3650008, 0.33846738])
expected_mean2 = np.array([5.56656896, 6.59117289, 4.91921459, 5.3213033, 6.68196096])

expected_var2 = np.array([0.22245441, 0.20320591, 0.22035304, 0.22053106, 0.20412299])

# gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_mean_var_misc_list = [
    [gp_emulator1_e, False, simulator1, exp_data1, expected_mean1, expected_var1],
    [gp_emulator1_e, True, simulator1, exp_data1, expected_mean1, expected_var1],
    [gp_emulator2_e, False, simulator2, exp_data2, expected_mean2, expected_var2],
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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    misc_data.theta_vals = theta_vals  # Set misc thetas
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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    candidate.theta_vals = theta_vals
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
eval_gp_mean_var_err_list = [gp_emulator1_e]


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
            0
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test(
            "False"
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand(
            1
        )  # Calc mean, var of gp
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(
            misc_data, feat_misc_data, "True"
        )  # Calc mean, var of gp


# This test function tests whether eval_gp_sse_var_test checker works correctly
expected_mean1_test_sse = np.array([0.6569498, 0.59890417])
expected_var1_test_sse = np.array([3.04214799, 3.53123217])
expected_mean2_test_sse = np.array([0.67082118])
expected_var2_test_sse = np.array([1.67899619])
# gp_emulator, exp_data, method, expected_mean, expected_var
eval_gp_sse_var_test_list = [
    [
        gp_emulator1_e,
        False,
        exp_data1,
        method,
        expected_mean1_test_sse,
        expected_var1_test_sse,
    ],
    [
        gp_emulator1_e,
        True,
        exp_data1,
        method,
        expected_mean1_test_sse,
        expected_var1_test_sse,
    ],
    [
        gp_emulator1_e,
        False,
        exp_data1,
        method,
        expected_mean1_test_sse,
        expected_var1_test_sse,
    ],
    [
        gp_emulator2_e,
        False,
        exp_data2,
        method,
        expected_mean2_test_sse,
        expected_var2_test_sse,
    ],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, exp_data, method, expected_mean, expected_var",
    eval_gp_sse_var_test_list,
)
def test_eval_gp_sse_var_test(
    gp_emulator, covar, exp_data, method, expected_mean, expected_var
):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_emulator.test_data.gp_mean, gp_emulator.test_data.gp_var = (
        gp_emulator.eval_gp_mean_var_test()
    )  # Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_test(
        method, exp_data, covar
    )  # Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()

    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert (
            len(sse_mean) * mult_factor
            == len(test_data.theta_vals)
            == len(sse_var) * mult_factor
        )
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2
        assert sse_var.shape[0] == sse_var.shape[1]


# This test function tests whether eval_gp_sse_var_val checker works correctly
expected_mean1_val_sse = np.array(
    [
        0.93710109,
        8.49797484,
        0.71889059,
        16.98973767,
        0.90409954,
        0.6569498,
        0.86899933,
        2.24827222,
        0.59890417,
        6.72762889,
    ]
)
expected_var1_val_sse = np.array(
    [
        2.71034979,
        14.2712367,
        2.34184944,
        27.77509619,
        2.63479837,
        3.04214799,
        2.54636151,
        4.66774442,
        3.53123217,
        11.42432879,
    ]
)
expected_mean2_val_sse = np.array(
    [3.05269292, 12.71297925, 0.36413124, 0.67082118, 0.54073811]
)
expected_var2_val_sse = np.array(
    [3.4359268, 12.34014306, 0.79408372, 1.67899619, 0.93497383]
)
# gp_emulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_val_list = [
    [gp_emulator1_e, False, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
    [gp_emulator1_e, True, exp_data1, expected_mean1_val_sse, expected_var1_val_sse],
    [gp_emulator2_e, False, exp_data2, expected_mean2_val_sse, expected_var2_val_sse],
]


@pytest.mark.parametrize(
    "gp_emulator, covar, exp_data, expected_mean, expected_var",
    eval_gp_sse_var_val_list,
)
def test_eval_gp_sse_var_val(gp_emulator, covar, exp_data, expected_mean, expected_var):
    train_data, test_data = gp_emulator.set_train_test_data(sep_fact, seed)
    gp_emulator.train_gp()
    gp_emulator.gp_val_data.gp_mean, gp_emulator.gp_val_data.gp_var = (
        gp_emulator.eval_gp_mean_var_val()
    )  # Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_val(
        method, exp_data, covar
    )  # Calc mean, var of gp sse
    mult_factor = exp_data.get_num_x_vals()
    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert (
            len(sse_mean) * mult_factor
            == len(gp_emulator.gp_val_data.theta_vals)
            == len(sse_var) * mult_factor
        )
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
    # Otherwise check that no covariance matrix is returned
    else:
        assert sse_var == None

# This test function tests whether eval_gp_sse_var_misc/cand checker works correctly
expected_mean1_sse = np.array([0.93710109])
expected_var1_sse = np.array([2.71034979])
expected_mean2_sse = np.array([3.05269292])
expected_var2_sse = np.array([3.4359268])

# gp_emulator, simulator, exp_data, expected_mean, expected_var
eval_gp_sse_var_misc_list = [
    [
        gp_emulator1_e,
        False,
        simulator1,
        exp_data1,
        expected_mean1_sse,
        expected_var1_sse,
    ],
    [
        gp_emulator1_e,
        True,
        simulator1,
        exp_data1,
        expected_mean1_sse,
        expected_var1_sse,
    ],
    [
        gp_emulator2_e,
        False,
        simulator2,
        exp_data2,
        expected_mean2_sse,
        expected_var2_sse,
    ],
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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    misc_data.theta_vals = theta_vals  # Set misc thetas
    feature_misc_data = gp_emulator.featurize_data(misc_data)  # Set feature vals
    misc_data.gp_mean, misc_data.gp_var = gp_emulator.eval_gp_mean_var_misc(
        misc_data, feature_misc_data
    )  # Calc mean, var of gp
    sse_mean, sse_var = gp_emulator.eval_gp_sse_var_misc(
        misc_data, method, exp_data, covar
    )  # Calc mean, var of gp sse

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
    theta = gp_emulator.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    candidate.theta_vals = theta_vals
    gp_emulator.cand_data = candidate  # Set candidate point
    gp_emulator.feature_cand_data = gp_emulator.featurize_data(
        gp_emulator.cand_data
    )  # Set feature vals
    gp_emulator.cand_data.gp_mean, gp_emulator.cand_data.gp_var = (
        gp_emulator.eval_gp_mean_var_cand()
    )  # Calc mean, var of gp
    sse_mean_cand, sse_var_cand = gp_emulator.eval_gp_sse_var_cand(
        method, exp_data, covar
    )  # Calc mean, var of gp sse

    mult_factor = exp_data.get_num_x_vals()

    assert np.allclose(sse_mean, expected_mean, rtol=1e-02)
    assert np.allclose(sse_mean_cand, expected_mean, rtol=1e-02)

    # If covar is false, check variance values are correct
    if covar == False:
        assert (
            len(sse_mean) * mult_factor
            == len(misc_data.theta_vals)
            == len(sse_var) * mult_factor
            == len(sse_mean_cand) * mult_factor
            == len(sse_var_cand) * mult_factor
        )
        assert np.allclose(sse_var, expected_var, rtol=1e-02)
        assert np.allclose(sse_var_cand, expected_var, rtol=1e-02)
    # Otherwise check that square covariance matrix is returned
    else:
        assert len(sse_var.shape) == 2
        assert len(sse_var_cand.shape) == 2
        assert sse_var.shape[0] == sse_var.shape[1]
        assert sse_var_cand.shape[0] == sse_var_cand.shape[1]


# This function tests whether eval_gp_sse_var_test/val/cand/misc throw the correct errors
# gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method, set_covar
eval_gp_sse_var_err_list = [
    [gp_emulator1_e, simulator1, exp_data1, False, True, True, method, True],
    [gp_emulator1_e, simulator1, exp_data1, True, False, True, method, True],
    [gp_emulator1_e, simulator1, exp_data1, True, True, False, method, True],
    [gp_emulator1_e, simulator1, exp_data1, True, True, False, method, False],
    [gp_emulator1_e, simulator1, exp_data1, True, True, False, None, True],
]


@pytest.mark.parametrize(
    "gp_emulator, simulator, exp_data, set_data, set_gp_mean, set_gp_var, method, set_covar",
    eval_gp_sse_var_err_list,
)
def test_eval_gp_sse_var_err(
    gp_emulator,
    simulator,
    exp_data,
    set_data,
    set_gp_mean,
    set_gp_var,
    method,
    set_covar,
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
    theta = gp_emulator_fail.gp_val_data.theta_vals[0].reshape(
        1, -1
    )  # Set "candidate thetas"
    theta_vals = np.repeat(theta.reshape(1, -1), exp_data.get_num_x_vals(), axis=0)
    candidate.theta_vals = theta_vals
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
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val(
                method, exp_data, "False"
            )
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_val(method, exp_data)
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_test()
        if set_data is False:
            gp_emulator_fail.test_data = None
        if set_gp_mean is False:
            gp_emulator_fail.test_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.test_data.gp_var = None
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test(
                method, exp_data, 0
            )
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_test(method, exp_data)
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_cand()
        if set_data is False:
            gp_emulator_fail.cand_data = None
        if set_gp_mean is False:
            gp_emulator_fail.cand_data.gp_mean = None
        if set_gp_var is False:
            gp_emulator_fail.cand_data.gp_var = None
        if set_covar is False:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand(
                method, exp_data, "True"
            )
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_cand(method, exp_data)
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
                misc_data, method, exp_data, 1
            )
        else:
            sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(
                misc_data, method, exp_data
            )
    with pytest.raises((AssertionError, AttributeError, ValueError)):
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(
            misc_data, feat_misc_data
        )  # Calc mean, var of gp
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(
            misc_data, method, None
        )

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        misc_data = candidate
        feat_misc_data = gp_emulator_fail.feature_cand_data
        gp_mean, gp_var = gp_emulator_fail.eval_gp_mean_var_misc(
            misc_data, feat_misc_data
        )  # Calc mean, var of gp
        sse_mean, sse_var = gp_emulator_fail.eval_gp_sse_var_misc(
            misc_data, method, exp_data, None
        )


# Test that add_next_theta_to_train_data(theta_best_sse_data) works correctly
# gp_emulator, simulator, exp_data, expected_mean, expected_var
add_next_theta_to_train_data_list = [
    [gp_emulator1_e, simulator1, exp_data1],
    [gp_emulator2_e, simulator2, exp_data2],
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

    # Append training data
    gp_emulator.add_next_theta_to_train_data(theta_best_data)

    assert (
        len(gp_emulator.train_data.theta_vals)
        == theta_before + exp_data.get_num_x_vals()
    )


# Test that add_next_theta_to_train_data(theta_best_data) throws correct errors correctly
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
# Test that add_next_theta_to_train_data(theta_best_data) works correctly
# gp_emulator, simulator, exp_data, bad_new_data, bad_train_data
add_next_theta_to_train_data_list = [
    [gp_emulator1_e, simulator1, exp_data1, None, train_data1],
    [gp_emulator1_e, simulator1, exp_data1, None, train_data2],
    [gp_emulator1_e, simulator1, exp_data1, None, train_data3],
    [gp_emulator1_e, simulator1, exp_data1, theta_best_data1, None],
    [gp_emulator1_e, simulator1, exp_data1, theta_best_data2, None],
    [gp_emulator1_e, simulator1, exp_data1, theta_best_data3, None],
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

    if bad_new_data is not None:
        theta_best_data = bad_new_data

    if bad_train_data is not None:
        gp_emulator_fail.train_data = bad_train_data

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        # Append training data
        gp_emulator_fail.add_next_theta_to_train_data(theta_best_data)
