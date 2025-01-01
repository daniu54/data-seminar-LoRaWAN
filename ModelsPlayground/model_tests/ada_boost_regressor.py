import numpy as np
import os
import uuid

# models
from scipy.optimize import curve_fit # multip. linear regression
from sklearn.svm import SVR # support vector forrest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import pandas as pd

import pipeline

data_path = pipeline.DEFAULT_DATA_FILE
data_path = ".." / data_path

data = pipeline.load_data(data_path)
data = pipeline.clean_data(data)
data = pipeline.add_features(data)

print(data.columns)
print(pipeline.calculate_correlation(data))


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_boost = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=14),
    n_estimators=484,
    learning_rate=0.6903938978839036
)


test_specifications = [
    {
        'id': f"optimal_ada_boost_{1}",
        # 'model': "models/optimal_ada_boost_1.joblib",
        'model': ada_boost,
        # 'skip_fitting': True,
        # "skip_cross_validation": False
        'verbose': 10,
        'n_jobs': -1,
        'save_model': True,
    },
]

pipeline.test_models(data, test_specifications)