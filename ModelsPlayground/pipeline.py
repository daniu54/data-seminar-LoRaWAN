from typing import Union, Callable

import os
import numpy as np

import uuid

# models
from scipy.optimize import curve_fit  # multip. linear regression
from sklearn.svm import SVR  # support vector forrest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from datetime import timedelta

import pandas as pd
import time

import joblib  # for persisting models

from enum import Enum

from pathlib import Path

DEFAULT_FEATURES = [
    "distance",
    "c_walls",
    "co2",
    "humidity",
    "pm25",
    "pressure",
    "temperature",
    "snr",
]

DEFAULT_STATS_FILE = Path("./results/model_stats.json")


def load_model(test_specification_row: pd.Series):
    """
    Load a model from a file and set the model arguments.
    """

    model_file = test_specification_row["model"]

    assert not pd.isna(model_file), "'model' is not defined in test specification row"
    assert isinstance(
        model_file, str
    ), f"expected 'model' to be a string, instead got {model_file}"

    print(f"Loading model from {model_file}")

    model = joblib.load(model_file)

    return model


def new_test_specification(model):
    """
    Create a new test specification from a SupportedModels enum value.
    """
    return {
        "model": model,  # model object or path string to load a model from
        "id": None,  # id for the test run, string, default: generate random id during testing
        "test_size": None,  # see train_test_split(test_size=test_size)
        "sample_size": None,  # how many data entries to use as integer, default: None, use 100% of provided data
        "features": None,  # array of column names from data, default: use DEFAULT_FEATURES
        "save_model": False,  # whether the model should be saved, default: False
        "random_state": None,  # used in data.sample and train_test_split, default: None
        "verbose": 0,  # whether to print verbose output from all functions that accept it, default: 0, no verbose output
        "n_jobs": -1,  # number of jobs to run in parallel, default: -1, use all available processors
    }


def get_model_file_name(test_specification: pd.Series):
    return f"models/{test_specification['id']}.joblib"


def set_defaults_for_column(
    test_specifications: pd.DataFrame,
    column: str,
    default: Union[Callable, object] = None,
):
    """
    Set default values for a column in test specifications.
    The default is an object or a callable that generates the default value.
    """
    if default is None:
        default = np.NaN

    if column not in test_specifications:
        test_specifications[column] = np.NaN

    for index, _ in test_specifications.iterrows():
        # overwrite with default only if the value is NaN
        if pd.isna(test_specifications.at[index, column]):
            default_value = default() if callable(default) else default

            # lists and dicts may be interpreted incorrectly by pandas, so we need to explicitly specify the column dtype as object
            # additionally, we correct string and bool columns to correct dtypes
            # only do it at the first iteration to avoid unnecessary overhead
            if index == 0:
                if isinstance(default_value, (list, dict)):
                    test_specifications[column] = test_specifications[column].astype(
                        object
                    )
                if isinstance(default_value, (str)):
                    test_specifications[column] = test_specifications[column].astype(
                        "string"
                    )
                if isinstance(default_value, (bool)):
                    test_specifications[column] = test_specifications[column].astype(
                        "bool"
                    )

            test_specifications.at[index, column] = default_value


def set_defaults_where_needed(test_specifications: pd.DataFrame, data: pd.DataFrame):
    """
    Set default values where needed in test specifications.
    """

    def generate_id():
        return str(uuid.uuid4()).replace("-", "")[:8]

    set_defaults_for_column(test_specifications, "id", generate_id)

    set_defaults_for_column(test_specifications, "features", default=DEFAULT_FEATURES)
    set_defaults_for_column(test_specifications, "sample_size", default=len(data))
    set_defaults_for_column(test_specifications, "save_model", default=False)
    set_defaults_for_column(test_specifications, "test_size", default=0.25)
    set_defaults_for_column(test_specifications, "random_state")
    set_defaults_for_column(test_specifications, "verbose", default=0)
    set_defaults_for_column(test_specifications, "n_jobs", default=-1)


def get_value(test_specification: pd.Series, column: str):
    """
    Get the value from a test specification column or None.
    """
    result = None

    if column in test_specification:
        result = test_specification[column]

        if isinstance(result, list):
            return result
        if pd.isna(result):
            return None

    return result


def test_models(
    data: pd.DataFrame,
    test_specifications: pd.DataFrame | list[object],
    stats_file: str = DEFAULT_STATS_FILE,
):
    """
    Test models with given data and test specifications.

    Parameters:
        data (pd.DataFrame): Data to test models on
        test_specifications (pd.DataFrame | list[object]): Test specifications for models, can be a DataFrame or a list of objects (for expected structure see new_test_specification)
    """

    stats_file = Path(stats_file)

    if isinstance(test_specifications, list):
        test_specifications = pd.DataFrame(test_specifications)
    else:
        # copy the DataFrame to avoid modifying the original
        test_specifications = test_specifications.copy()

    assert (
        "model" in test_specifications
    ), "'model' column is required in test specifications"

    set_defaults_where_needed(test_specifications, data)

    results = test_specifications.copy()

    # move columns around for readability
    results.insert(0, "id", results.pop("id"))
    results.insert(1, "model", results.pop("model"))
    results.insert(2, "mse", np.NaN)
    results.insert(3, "r2", np.NaN)
    results.insert(4, "model_parameters", np.NaN)
    results["model_parameters"] = results["model_parameters"].astype(
        object
    )  # need to tell pandas that we will be storing dictionaries here

    test_start = time.time()

    for index, test in test_specifications.iterrows():
        id = get_value(test, "id")
        test_size = get_value(test, "test_size")
        features = get_value(test, "features")
        random_state = get_value(test, "random_state")
        sample_size = get_value(test, "sample_size")
        save_model = get_value(test, "save_model")
        verbose = int(get_value(test, "verbose"))
        n_jobs = int(get_value(test, "n_jobs"))

        model = test["model"]

        if isinstance(model, str):
            model = load_model(test)
            results.at[index, "model"] = f"{model} (loaded from {test['model']})"
        else:
            print(f"Using provided model {model}")
            pass

        if hasattr(model, "random_state"):
            model.set_params(random_state=random_state)

        if hasattr(model, "verbose"):
            model.set_params(verbose=verbose)

        if hasattr(model, "n_jobs"):
            model.set_params(n_jobs=n_jobs)

        data_sampled = data.sample(n=int(sample_size), random_state=random_state)

        x = data_sampled[features]
        y = data_sampled["exp_pl"]

        print(
            f"Test {index + 1} of {len(test_specifications)} with id {id} started at {pd.Timestamp.now()}"
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        start_time = time.time()
        test_start_time = time.time()

        print(
            f"Fitting model {model} for train size {len(x_test)} (test_size={test_size}) started at {pd.Timestamp.now()}"
        )
        start_time = time.time()
        model.fit(x_train, y_train)
        time_fitting = time.time() - start_time

        print(
            f"Predicting model {model} for test size {len(x_test)} (test_size={test_size}) started at {pd.Timestamp.now()}"
        )
        start_time = time.time()
        y_test_pred = model.predict(x_test)
        time_pred = time.time() - start_time

        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        # save results
        results.at[index, "model"] = str(model)
        results.at[index, "model_type"] = model.__class__.__name__
        results.at[index, "model_type_full"] = (
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        )
        results.at[index, "time_fitting"] = str(timedelta(seconds=time_fitting))
        results.at[index, "time_pred"] = str(timedelta(seconds=time_pred))
        results.at[index, "mse"] = mse
        results.at[index, "r2"] = r2
        results.at[index, "model_parameters"] = model.get_params()

        # overfitting analysis
        print(
            f"Calculation of overfitting metrics for model {model} for test size {len(x_test)} (test_size={test_size}) started at {pd.Timestamp.now()}"
        )
        y_train_pred = model.predict(x_train)

        # performance on training data
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        results.at[index, "mse_train"] = mse_train
        results.at[index, "r2_train"] = r2_train
        results.at[index, "mse_diff_train_test"] = mse_train - mse
        results.at[index, "r2_diff_train_test"] = r2_train - r2

        # cross-validation
        folds = 5
        print(
            f"Calculation of cross validation metrics ({folds} folds) for model {model} over whole dataset"
        )

        print(
            f"Calculation of cross validation for mse started at {pd.Timestamp.now()}"
        )
        start_time = time.time()
        cv_mse = -cross_val_score(
            model,
            x,
            y,
            cv=folds,
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
            verbose=verbose,
        )
        cross_val_mse_time = str(timedelta(seconds=time.time() - start_time))
        print(f"Calculation of cross validation for mse took {cross_val_mse_time}")

        print(f"Calculation of cross validation for r2 started at {pd.Timestamp.now()}")
        start_time = time.time()
        cv_r2 = cross_val_score(
            model, x, y, cv=folds, scoring="r2", n_jobs=n_jobs, verbose=verbose
        )
        cross_val_r2_time = str(timedelta(seconds=time.time() - start_time))
        print(f"Calculation of cross validation for r2 took {cross_val_r2_time}")

        cross_val_colum_name = f"cross_validation_k{folds}"
        cross_val_mse_column = cross_val_colum_name + "_mse"
        cross_val_r2_column = cross_val_colum_name + "_r2"

        # need to tell pandas that we will be storing lists in these columns
        for column in [cross_val_mse_column, cross_val_r2_column]:
            if column not in results:
                results[column] = np.NaN
                results[column] = results[column].astype(object)

        results.at[index, cross_val_mse_column] = cv_mse
        results.at[index, cross_val_r2_column] = cv_r2

        results.at[index, cross_val_mse_column + "_mean"] = cv_mse.mean()
        results.at[index, cross_val_r2_column + "_mean"] = cv_r2.mean()

        results.at[index, "time_" + cross_val_mse_column] = cross_val_mse_time
        results.at[index, "time_" + cross_val_r2_column] = cross_val_r2_time

        results.at[index, "date_ran"] = str(pd.Timestamp.now())
        results.at[index, "time_test_run"] = str(
            timedelta(seconds=time.time() - test_start_time)
        )

        # save the model to a file if needed
        if not pd.isna(save_model) and save_model:
            model_file = get_model_file_name(test)
            print(f"Saving model to {model_file}")
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            joblib.dump(model, model_file)

        # save test results for each iteration
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        try:
            print(f"Saving model stats to {stats_file}")
            with open(stats_file, "x") as f:
                results.loc[[index]].to_json(f, orient="records", lines=True)
        except FileExistsError:
            with open(stats_file, "a") as f:
                results.loc[[index]].to_json(f, orient="records", lines=True)

        print(
            f"Test {index + 1} of {len(test_specifications)} with id {id} ended at {pd.Timestamp.now()}"
        )

    test_end = time.time()

    print(
        f"Test took {time.strftime('%H:%M:%S', time.gmtime(test_end - test_start))} for {len(test_specifications)} tests"
    )

    return results
