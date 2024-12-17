import os
import numpy as np

import uuid

# models
from scipy.optimize import curve_fit # multip. linear regression
from sklearn.svm import SVR # support vector forrest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from datetime import timedelta

import pandas as pd
import time

import joblib # for persisting models

from enum import Enum

DEFAULT_FEATURES = ['distance', 'c_walls', 'co2', 'humidity', 'pm25', 'pressure', 'temperature', 'snr']
DEFAULT_STATS_FILE = "results/stats.csv"

def load_model(model_file: str):
    """
    Load a model from a file and set the model arguments.
    """

    print(f"Loading model from {model_file}")

    model = joblib.load(model_file)

    return model

def new_test_specification(model):
    """
    Create a new test specification from a SupportedModels enum value.
    """
    return {
        'model': model, # model object or path string to load a model from
        'id': None, # id for the test run, string, default: generate random id during testing
        'test_size': None, # see train_test_split(test_size=test_size)
        'sample_size': None, # how many data entries to use as integer, default: None, use 100% of provided data
        'features': None, # array of column names from data, default: use DEFAULT_FEATURES
        'save_model': False, # whether the model should be saved, default: False
        'random_state': None, # used in data.sample and train_test_split, default: None
    }

def get_model_file_name(test_specification: pd.Series):
    return f"models/{test_specification['id']}.joblib"

def set_defaults_for_column(test_specifications: pd.DataFrame, column: str, default: callable | object = np.NaN):
    """
    Set default values for a column in test specifications.
    The default is a callable or an object that generates the value for a given row index and row.
    If default is None and the column is not in test_specifications, the column is set to NaN.
    """
    if default is None and column not in test_specifications:
        test_specifications[column] = default
        return

    for index, row in test_specifications.iterrows():
        if column not in row or pd.isna(row[column]):
            # retrieve the default value from callable if default is a function
            default_value = default(index) if callable(default) else default

            # lists and dicts may be interpreted as scalar values by pandas, so we need to explicitly specify the column dtype to object
            if isinstance(default_value, (list, dict)):
                # do it only once for the first row to avoid performance issues
                if index == 0:
                    test_specifications[column] = test_specifications[column].astype(object)

            test_specifications.at[index, column] = default_value

def set_defaults_where_needed(test_specifications: pd.DataFrame, data: pd.DataFrame):
    """
    Set default values where needed in test specifications.
    """
    def generate_id(_):
        return str(uuid.uuid4()).replace('-', '')[:8]

    set_defaults_for_column(test_specifications, 'id', generate_id)

    set_defaults_for_column(test_specifications, 'features', default=DEFAULT_FEATURES)
    set_defaults_for_column(test_specifications, 'sample_size', default=len(data))
    set_defaults_for_column(test_specifications, 'save_model', default=False)
    set_defaults_for_column(test_specifications, 'test_size', default=0.25)
    set_defaults_for_column(test_specifications, 'random_state')

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

def test_models(data: pd.DataFrame, test_specifications: pd.DataFrame | list[object], stats_file: str = DEFAULT_STATS_FILE):
    """
    Test models with given data and test specifications.

    Parameters:
        data (pd.DataFrame): Data to test models on
        test_specifications (pd.DataFrame | list[object]): Test specifications for models, can be a DataFrame or a list of objects (for expected structure see new_test_specification)
    """

    if isinstance(test_specifications, list):
        test_specifications = pd.DataFrame(test_specifications)
    else:
        # copy the DataFrame to avoid modifying the original
        test_specifications = test_specifications.copy()

    assert 'model' in test_specifications, "'model' column is required in test specifications"

    set_defaults_where_needed(test_specifications, data)

    results = test_specifications.copy()

    # move columns around for readability
    results.insert(0, 'id', results.pop('id'))
    results.insert(1, 'model', results.pop('model'))
    results.insert(2, 'mse', np.NaN)
    results.insert(3, 'r2', np.NaN)

    test_start = time.time()

    for index, test in test_specifications.iterrows():
        id = get_value(test, 'id')
        test_size = get_value(test, 'test_size')
        features = get_value(test, 'features')
        random_state = get_value(test, 'random_state')
        sample_size = get_value(test, 'sample_size')
        save_model = get_value(test, 'save_model')

        model = test['model']
        assert not pd.isna(test['model']), f"'model' is defined in test specification at index {index} but is NaN"

        data_sampled = data.sample(n=int(sample_size), random_state=random_state)

        x = data_sampled[features]
        y = data_sampled['exp_pl']

        print(f"Test {index + 1} of {len(test_specifications)} with id {id}")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        
        start_time = time.time()
        if isinstance(model, str):
            model = load_model(model)
            results.at[index, 'model'] = f"{model} (loaded from {test['model']})"
        else:
            print(f"Using provided model {model}")
            pass

        print(f"Fitting model {model} for train size {len(x_train)}")
        start_time = time.time()
        model.fit(x_train, y_train)
        time_fitting = time.time() - start_time

        print(f"Predicting model {model} for test size {len(x_test)}")
        start_time = time.time()
        y_pred = model.predict(x_test)
        time_pred = time.time() - start_time

        results.at[index, 'time_fitting'] = str(timedelta(seconds=time_fitting))
        results.at[index, 'time_pred'] = str(timedelta(seconds=time_pred))
        results.at[index, 'mse'] = mean_squared_error(y_test, y_pred)
        results.at[index, 'r2'] = r2_score(y_test, y_pred)
        results.at[index, 'model_hyperparameters'] = model.get_params()

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
                results.loc[[index]].to_csv(f, index=False)
        except FileExistsError:
            with open(stats_file, "a") as f:
                results.loc[[index]].to_csv(f, header=False, index=False) 

    test_end = time.time()

    print(f"Test took {time.strftime('%H:%M:%S', time.gmtime(test_end - test_start))} for {len(test_specifications)} tests")

    return results