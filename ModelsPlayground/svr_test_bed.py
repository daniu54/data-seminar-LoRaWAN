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

