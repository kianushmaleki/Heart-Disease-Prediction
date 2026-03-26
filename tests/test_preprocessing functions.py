import pandas as pd
import numpy as np
import pytest
import sys
import os
import yaml
from typing import Tuple, Any


# 1. Get the directory where test_model.py lives
current_dir = os.path.dirname(__file__)

# 2. Go up one level to the project root, then into 'src'
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))

# 3. Add to sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

# 4. Now this will work
from model_training import load_data

X_train, y_train, X_test, y_test = load_data("configs/configs.yaml")
print(len(X_train), len(y_train), len(X_test), len(y_test))

# this is a test to check that the load_data function correctly loads the data and that there are no null values in the loaded data
def test_load_data_no_nulls():
    assert not X_train.isnull().values.any(), "X_train contains null values"
    assert not y_train.isnull().values.any(), "y_train contains null values"
    assert not X_test.isnull().values.any(), "X_test contains null values"
    assert not y_test.isnull().values.any(), "y_test contains null values"
