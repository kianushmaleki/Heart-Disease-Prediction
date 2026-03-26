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

def test_check_column_names():
    
    # This is the exact list of columns your model needs
    expected_columns = [
        'age', 'trestbps', 'thalch', 'oldpeak', 'sex_Male', 
        'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina', 
        'fbs_True', 'restecg_normal', 'restecg_st-t abnormality', 
        'exang_True', 'slope_flat', 'slope_upsloping'
    ]
    
    # Get the actual columns from your loaded training data
    actual_columns = list(X_train.columns)
    
    # This single line checks if the lists match perfectly
    assert actual_columns == expected_columns, f"Expected {expected_columns} but got {actual_columns}"