# 1. Standard Library Imports
import os
import yaml
import json
from typing import Tuple, Any

# 2. Data Manipulation & Numerical Libraries
import numpy as np
import pandas as pd

# 3. Scikit-Learn: Preprocessing & Model Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# 4. Scikit-Learn: Models (Classifiers)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 5. Scikit-Learn: Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 6. Experiment Tracking
import mlflow
import mlflow.sklearn

# 7. Custom Modules
from data_processing import load_data, clean_data, save_data, split_data
from model_training import load_model_params, load_data, train_model


def run_experiment(config):
    with open(config, 'r') as file:
        config = yaml.safe_load(file)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student-dropout-prediction")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ── Log all configuration as parameters ──
        if config['model']['type'] == "logistic_regression":
            mlflow.log_param("model_type", "Logistic Regression")
            mlflow.log_param("lr_C", config['model']['params']['lr_C'])
        
        elif config['model']['type'] == "random_forest":
            mlflow.log_param("model_type", "Random Forest")
            mlflow.log_param("rf_n_estimators", config['model']['params']['rf_n_estimators'])
            mlflow.log_param("rf_max_depth", config['model']['params']['rf_max_depth'])
            mlflow.log_param("rf_min_samples_split", config['model']['params']['rf_min_samples_split'])
        
        elif config['model']['type'] == "gradient_boosting":
            mlflow.log_param("model_type", "Gradient Boosting")
            mlflow.log_param("gb_n_estimators", config['model']['params']['gb_n_estimators'])
            mlflow.log_param("gb_learning_rate", config['model']['params']['gb_learning_rate'])
            mlflow.log_param("gb_max_depth", config['model']['params']['gb_max_depth'])
            mlflow.log_param("gb_min_samples_split", config['model']['params']['gb_min_samples_split'])
            mlflow.log_param("gb_min_samples_leaf", config['model']['params']['gb_min_samples_leaf'])
        
        else:
            raise ValueError(f"Unsupported model type: {config['model']['type']}")

        # ── Train ──
        if 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
            X_train, y_train, X_test, y_test = load_data('configs/configs.yaml')

        model_info = train_model('configs/configs.yaml')
        mlflow.log_metric("accuracy", model_info["accuracy"])
        mlflow.log_metric("precision", model_info["precision"])
        mlflow.log_metric("recall", model_info["recall"])
        mlflow.log_metric("f1_score", model_info["f1_score"])
        mlflow.log_metric("auc_roc", model_info["auc_roc"])
        mlflow.sklearn.log_model(model_info["model"], name = "model")  
    
    return run_id


print(run_experiment('configs/configs.yaml'))