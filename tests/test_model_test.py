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
from model_training import train_model


def test_train_model():
    config_file = "configs/configs.yaml"
    train_model("test")


