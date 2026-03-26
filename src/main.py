from sklearn.linear_model import LogisticRegression
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder




import mlflow


def run_experiment(config):
    mlflow.set_experiment("student-dropout-prediction")
    with mlflow.start_run():