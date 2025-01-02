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

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_boost = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=14),
    n_estimators=484,
    learning_rate=0.6903938978839036,
)

from xgboost import XGBRegressor

xg_boost = XGBRegressor(
    max_depth=15,
    learning_rate=0.019501791550458624,
    n_estimators=436,
    gamma=2.5671387248762607,
    min_child_weight=5,
    subsample=0.5728919054154519,
    colsample_bytree=0.889193793406033,
    reg_alpha=1.4832156197353648,
    reg_lambda=5.336603614464974,
)


from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting_regressor = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
)

skip_on_duplicate = True
save_model = True

# TODO fix features specification
test_specifications = [
    {
        "id": f"ada_boost_{1}",
        "model": ada_boost,
        "verbose": 10,
        "n_jobs": -1,
        "save_model": save_model,
        "skip_on_duplicate": skip_on_duplicate,
    },
    {
        "id": f"xg_boost_{1}",
        "model": xg_boost,
        "verbose": 10,
        "n_jobs": -1,
        "save_model": save_model,
        "skip_on_duplicate": skip_on_duplicate,
    },
    {
        "id": f"gradient_boosting_regressor_{1}",
        "model": gradient_boosting_regressor,
        "verbose": 10,
        "n_jobs": -1,
        "save_model": save_model,
        "skip_on_duplicate": skip_on_duplicate,
    },
]

pipeline.test_models(data, test_specifications, stats_file=stats_file)
