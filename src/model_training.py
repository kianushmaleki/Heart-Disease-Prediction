from sklearn.linear_model import LogisticRegression
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Any
import mlflow



# 1. Updated Type Hint to reflect the 8 the model
def load_model_params(config_file: str):
    print(f"Loading configuration of the model from {config_file}...")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    if config['model']['type'] == "logistic_regression":
        print("Model type set to Logistic Regression.")
        return LogisticRegression(
                C = config['model']['params']['lr_C'], 
                random_state=config['model']['params']['random_state']
            )  
    
    elif config['model']['type'] == "random_forest":
        print("Model type set to Random Forest.")
        return RandomForestClassifier(
                n_estimators=config['model']['params']['rf_n_estimators'], 
                max_depth=config['model']['params']['rf_max_depth'], 
                min_samples_split=config['model']['params']['rf_min_samples_split'],
                random_state=config['model']['params']['random_state']
            )
    
    elif config['model']['type'] == "gradient_boosting":
        print("Model type set to Gradient Boosting.")
        return GradientBoostingClassifier(
                n_estimators=config['model']['params']['gb_n_estimators'], 
                learning_rate=config['model']['params']['gb_learning_rate'], 
                max_depth=config['model']['params']['gb_max_depth'], 
                min_samples_split=config['model']['params']['gb_min_samples_split'], 
                min_samples_leaf=config['model']['params']['gb_min_samples_leaf'], 
                random_state=config['model']['params']['random_state']
            )
    
    else:
        raise ValueError(f"Unsupported model type: {config['model']['type']}")
        

def load_data(config_file: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    print(f"Loading the data from {config_file}...")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    print("Loading data...")
    # Using .squeeze() converts a single-column DataFrame into a Series automatically
    X_train = pd.read_csv(config['data']['train_X_path'])
    y_train = pd.read_csv(config['data']['train_y_path']).squeeze("columns")
    X_test = pd.read_csv(config['data']['test_X_path'])
    y_test = pd.read_csv(config['data']['test_y_path']).squeeze("columns")
    
    print("Data loaded successfully.")
    return X_train, y_train, X_test, y_test

def train_model(config_file: str):
    model = load_model_params(config_file)
    X_train, y_train, X_test, y_test = load_data(config_file)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob_all = model.predict_proba(X_test)


    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')
    
    # FIX: For ROC AUC with multiclass, use multi_class='ovr' (One-vs-Rest)
    auc = roc_auc_score(y_test, y_prob_all, multi_class='ovr')

    output = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc
    }
    return output


if __name__ == "__main__":
    print("Training the model...")
    print(train_model())