import numpy as np
import os
import uuid

import sys
from pathlib import Path

# models
import pandas as pd

# quick-and-dirty import, since I could not get relative imports to work
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str((root_dir).absolute()))
import pipeline

data_path = pipeline.DEFAULT_DATA_FILE
data_path = ".." / data_path

stats_file = pipeline.DEFAULT_STATS_FILE
stats_file = ".." / stats_file

data = pipeline.load_data(data_path)
data = pipeline.clean_data(data)
data = pipeline.add_features(data)

print(data.columns)
print(pipeline.calculate_correlation(data))

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

skip_on_duplicate = True
save_model = True

# TODO fix features specification
test_specifications = [
    {
        "id": "random_forrest_regressor_default",
        "model": RandomForestRegressor(),
        "save_model": True,
        "verbose": 10,
    },
    {
        "id": "mlp_regressor_default",
        "model": MLPRegressor(),
        "save_model": True,
        "verbose": 10,
    },
    {
        "id": "kneighbors_regressor_default",
        "model": KNeighborsRegressor(),
        "save_model": True,
        "verbose": 10,
    },
]

pipeline.test_models(data, test_specifications, stats_file=stats_file)
