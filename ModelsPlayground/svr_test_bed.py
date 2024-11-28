import os
import numpy as np

# models
from scipy.optimize import curve_fit # multip. linear regression
from sklearn.svm import SVR # support vector forrest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import time

def svr_test_bed(data: pd.DataFrame, test_specification: pd.DataFrame):
    method = "SVR"

    test_data = test_specification.copy()
    test_data['method'] = "SVR"

    # move columns id and method to the front for readability
    test_data.insert(0, 'id', test_data.pop('id'))
    test_data.insert(1, 'method', test_data.pop('method'))

    test_start = time.time()

    for index, test in test_data.iterrows():
        assert 'test_size' in test, "'test_size' is required in test specification"
        assert 'features' in test, "'features' is required in test specification"
        assert 'output_file' in test, "'output_file' is required in test specification"

        # best model
        best_model = None
        best_r2 = None
        best_id = None

        # defaults for test
        output_file = test['output_file']
        test_size = test['test_size']
        features = test['features']
        random_state = None
        data_sampled = data
        sample_size = len(data)
        model_arguments = {}
        kernel = 'rbf'
        c = 1.0
        epsilon = 0.1

        # overwrite defaults if specified in test_specification
        if 'random_state' in test:
            random_state = test['random_state']
            if pd.isna(random_state):
                random_state = None

        if 'sample_size' in test:
            sample_size = test['sample_size']
            sample_size = int(sample_size)

            data_sampled = data.sample(n=sample_size, random_state=random_state)
        else:
            data_sampled = data

        if 'model_arguments' in test:
            model_arguments = test['model_arguments']
            if 'kernel' in model_arguments:
                kernel = model_arguments['kernel']
            if 'C' in model_arguments:
                c = model_arguments['C']
            if 'epsilon' in model_arguments:
                epsilon = model_arguments['epsilon']

        x = data_sampled[features]
        y = data_sampled['exp_pl']

        print(f"Test {index + 1} of {len(test_data)} with sample_size {sample_size}, test_size {test_size} and features {features}")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        
        print("Creating Model")
        start_time = time.time()
        model = SVR(
            kernel=kernel,
            C=c,
            epsilon=epsilon
        )
        time_setup = time.time() - start_time

        print("Fitting Model")
        start_time = time.time()
        model.fit(x_train, y_train)
        time_fitting = time.time() - start_time

        print("Predicting")
        start_time = time.time()
        y_pred = model.predict(x_test)
        time_pred = time.time() - start_time

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        test_data.at[index, 'time_setup'] = time.strftime('%H:%M:%S', time.gmtime(time_setup))
        test_data.at[index, 'time_fitting'] = time.strftime('%H:%M:%S', time.gmtime(time_fitting))
        test_data.at[index, 'time_pred'] = time.strftime('%H:%M:%S', time.gmtime(time_pred))
        test_data.at[index, 'mse'] = mse
        test_data.at[index, 'r2'] = r2

        if best_r2 is None or r2 > best_r2:
            best_model = model
            best_r2 = r2
            best_id = test['id']
        
        # save test results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            with open(output_file, "x") as f:
                test_data.loc[[index]].to_csv(f, index=False)
        except FileExistsError:
            with open(output_file, "a") as f:
                test_data.loc[[index]].to_csv(f, header=False, index=False) 

    test_end = time.time()

    print(f"Test took {time.strftime('%H:%M:%S', time.gmtime(test_end - test_start))}")
    print("Best model id: ", best_id, " with r2: ", best_r2)

    return (test_data, best_model, best_id)