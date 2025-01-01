from typing import Union, Callable

import os
import sys
from io import StringIO
import numpy as np

import uuid

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from datetime import timedelta
import traceback

import pandas as pd

import joblib  # for persisting models

from pathlib import Path

DEFAULT_FEATURES = [
    # from provided data
    "distance",
    "co2",
    "humidity",
    "pm25",
    "pressure",
    "temperature",
    "frequency",
    "snr",
    # "c_walls",
    # "w_walls",

    # new features, see `add_features`
    "inverse_distance_squared",
    "humidity_temperature_product",
]

DEFAULT_STATS_FILE = Path("./results/model_stats.json")

DEFAULT_DATA_FILE = Path("aggregated_measurements_data.csv")

def load_data(data_path: Path = DEFAULT_DATA_FILE):
    """
    Load data from the file at data_path.
    """

    print(f"Loading data from {data_path}")

    data = pd.read_csv(data_path, index_col=0)

    return data

def clean_data(data: pd.DataFrame):
    """
    Clean the data.
    """

    print("Cleaning data by removing rows with NaN values")
    cleaned_data = data.dropna()

    print("Dropped ", len(data) - len(cleaned_data), " rows with NaN values")

    return cleaned_data

def add_features(data: pd.DataFrame):
    """
    Add additional features to the data.
    """

    distance = data["distance"].values
    humidity = data["humidity"].values
    temperature = data["temperature"].values

    inverse_distance_squared = 1 / (distance**2 + 1e-10)
    humidity_temperature_product = humidity * temperature  # Humidity × Temperature

    new_features = pd.DataFrame({
        "inverse_distance_squared": inverse_distance_squared,
        "humidity_temperature_product": humidity_temperature_product
    }, index=data.index)

    enhanced_data = pd.concat([data, new_features], axis=1)

    return enhanced_data

def calculate_correlation(data: pd.DataFrame):
    """
    Calculate the correlation matrix for the data.
    """

    print("Calculating correlation matrix for the data")

    correlation_matrix = data.corr()

    return correlation_matrix["exp_pl"]

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
    Create a new test specification with default values.
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
        "skip_cross_validation": False,  # whether to skip cross validation and recording these metrics, default: False
        "skip_fitting": False,  # whether to skip fitting the model and recording these metrics, default: False
        "skip_on_duplicate": False, # whether to skip the test if the test with the same id already exists in results file, default: False
    }


def get_model_file_name(test_specification: pd.Series):
    return f"models/{test_specification['id']}.joblib"


def get_model_logfile_name(test_specification: pd.Series, test_end: pd.Timestamp):
    timestamp_numeric = int(test_end.timestamp())
    return f"models/{test_specification['id']}_{timestamp_numeric}.logs"


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
                    # setting column type to bool autocasts NaN to True, which breaks this function
                    # for now, just set to column type object instead
                    test_specifications[column] = test_specifications[column].astype(
                        object
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
    set_defaults_for_column(test_specifications, "skip_cross_validation", default=False)
    set_defaults_for_column(test_specifications, "skip_fitting", default=False)
    set_defaults_for_column(test_specifications, "skip_on_duplicate", default=False)


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
    results.insert(5, "unused_features", np.NaN)
    results["unused_features"] = results["unused_features"].astype(
        object
    )  # need to tell pandas that we will be storing dictionaries here

    testing_start = pd.Timestamp.now()

    for index, test in test_specifications.iterrows():
        logs_buffer = StringIO()
        tee_stdout = TeeStdout(logs_buffer)
        tee_stderr = TeeStderr(logs_buffer)

        # capture output to a file and console
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr

        id = get_value(test, "id")
        test_size = get_value(test, "test_size")
        features = get_value(test, "features")
        random_state = get_value(test, "random_state")
        sample_size = get_value(test, "sample_size")
        save_model = get_value(test, "save_model")
        verbose = int(get_value(test, "verbose"))
        n_jobs = int(get_value(test, "n_jobs"))
        skip_cross_validation = get_value(test, "skip_cross_validation")
        skip_fitting = get_value(test, "skip_fitting")
        skip_on_duplicate = get_value(test, "skip_on_duplicate")

        if skip_on_duplicate:
            model_stats = pd.read_json(stats_file, orient='records', lines=True)
            ids = model_stats["id"].values
            if id in ids:
                print(f"Skipping test with id {id} because skip_on_duplicate is set to True and the test already exists in the results file")
                continue

        try:
            model = test["model"]

            if isinstance(model, str):
                model = load_model(test)
                results.at[index, "model"] = f"{model} (loaded from {test['model']})"
            else:
                print(f"Using provided model {model}")
                pass

            if "random_state" in model.get_params():
                model.set_params(random_state=random_state)

            if "verbose" in model.get_params():
                if isinstance(model.get_params()["verbose"], bool):
                    # some models require verbose to be a boolean
                    model.set_params(verbose=verbose > 0)
                else:  # assume verbose is an integer
                    model.set_params(verbose=verbose)

            if "n_jobs" in model.get_params():
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

            test_start = pd.Timestamp.now()

            results.at[index, "model"] = str(model)
            results.at[index, "model_type"] = model.__class__.__name__
            results.at[index, "model_type_full"] = (
                f"{model.__class__.__module__}.{model.__class__.__name__}"
            )
            results.at[index, "model_parameters"] = model.get_params()

            results.at[index, "time_test_start"] = test_start
            results.at[index, "time_test_start_pretty"] = str(test_start)

            results.at[index, "unused_features"] = list(set(data.columns) - set(features))

            if skip_fitting:
                print(
                    f"Skipping fitting for model {model} because skip_fitting is set to True"
                )
            else:
                print(
                    f"Fitting model {model} for train size {len(x_test)} (test_size={test_size}) started at {pd.Timestamp.now()}"
                )
                start_time = pd.Timestamp.now()
                model.fit(x_train, y_train)
                time_fitting = (pd.Timestamp.now() - start_time).total_seconds()
                results.at[index, "time_fitting"] = str(timedelta(seconds=time_fitting))

            # save the model to a file after fitting
            if save_model:
                model_file = get_model_file_name(test)
                print(f"Saving model to {model_file}")
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                joblib.dump(model, model_file)

            print(
                f"Predicting model {model} for test size {len(x_test)} (test_size={test_size}) started at {pd.Timestamp.now()}"
            )

            start_time = pd.Timestamp.now()
            y_test_pred = model.predict(x_test)
            time_pred = (pd.Timestamp.now() - start_time).total_seconds()

            results.at[index, "time_pred"] = str(timedelta(seconds=time_pred))

            mse = mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)

            results.at[index, "mse"] = mse
            results.at[index, "r2"] = r2

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
            if skip_cross_validation:
                print(
                    f"Skipping cross validation for model {model} because skip_cross_validation is set to True"
                )
            else:
                folds = 5
                print(
                    f"Calculation of cross validation metrics ({folds} folds) for model {model} over whole dataset"
                )

                print(
                    f"Calculation of cross validation for mse started at {pd.Timestamp.now()}"
                )
                start_time = pd.Timestamp.now()
                cv_mse = -cross_val_score(
                    model,
                    x,
                    y,
                    cv=folds,
                    scoring="neg_mean_squared_error",
                    # n_jobs cause google colab workers to hang for some values
                    # n_jobs=n_jobs,
                    verbose=verbose,
                )
                cross_val_mse_time = (pd.Timestamp.now() - start_time).total_seconds()
                cross_val_mse_time = str(timedelta(seconds=cross_val_mse_time))
                print(
                    f"Calculation of cross validation for mse took {cross_val_mse_time}"
                )

                print(
                    f"Calculation of cross validation for r2 started at {pd.Timestamp.now()}"
                )
                start_time = pd.Timestamp.now()
                cv_r2 = cross_val_score(
                    model,
                    x,
                    y,
                    cv=folds,
                    scoring="r2",
                    # n_jobs cause google colab workers to hang for some values
                    # n_jobs=n_jobs,
                    verbose=verbose,
                )
                cross_val_r2_time = (pd.Timestamp.now() - start_time).total_seconds()
                cross_val_r2_time = str(timedelta(seconds=cross_val_r2_time))
                print(
                    f"Calculation of cross validation for r2 took {cross_val_r2_time}"
                )

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

            results.at[index, "test_success"] = True

            print(
                f"Test {index + 1} of {len(test_specifications)} with id {id} ended at {pd.Timestamp.now()}"
            )

        except Exception as e:
            print(
                f"Test {index + 1} of {len(test_specifications)} failed with an error {e}"
            )
            traceback.print_exc()

            results.at[index, "test_success"] = False
        finally:
            test_end = pd.Timestamp.now()
            test_duration = (test_end - test_start).total_seconds()

            results.at[index, "time_test_end"] = test_end
            results.at[index, "time_test_end_pretty"] = str(test_end)
            results.at[index, "time_test_duration"] = str(
                timedelta(seconds=test_duration)
            )

            # save test results for each iteration
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            try:
                print(f"Saving model stats to {stats_file}")
                with open(stats_file, "x") as f:
                    results.loc[[index]].to_json(f, orient="records", lines=True)
            except FileExistsError:
                with open(stats_file, "a") as f:
                    f.write(
                        "\n"
                    )  # write a newline into file to avoid json parsing errors
                    results.loc[[index]].to_json(f, orient="records", lines=True)

            # Write logs to a file
            logs_file = get_model_logfile_name(test, test_end)
            with open(logs_file, "w") as f:
                print(f"Saving logs to {logs_file}")
                f.write(logs_buffer.getvalue())

            # Reset original stdout and stderr
            sys.stdout = tee_stdout.stdout
            sys.stderr = tee_stderr.stderr

            logs_buffer.close()

    testing_end = pd.Timestamp.now()

    testing_duration = (testing_end - testing_start).total_seconds()
    testing_duration = str(timedelta(seconds=testing_duration))

    print(f"Test took {testing_duration} for {len(test_specifications)} tests")

    return results


class TeeStdout:  # used to capture stdout and write it to a file
    def __init__(self, string_buffer: StringIO):
        self.stdout = sys.stdout
        self.buffer = string_buffer

    def write(self, data):
        # write to both console and the file
        self.buffer.write(data)
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()


class TeeStderr:  # used to capture stdout and write it to a file
    def __init__(self, string_buffer: StringIO):
        self.stderr = sys.stderr
        self.buffer = string_buffer

    def write(self, data):
        self.buffer.write(data)
        self.stderr.write(data)

    def flush(self):
        self.stderr.flush()
