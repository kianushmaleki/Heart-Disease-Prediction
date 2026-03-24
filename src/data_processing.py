import numpy as np
import pandas as pd

def load_data(file_path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    



def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Parameters:
    data (pd.DataFrame): The raw data as a pandas DataFrame.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    # Handle missing values: There are three columns with most missing values: "ca", "thal", and "chol". We do not use these columns in our model, so we drop them.
    data = data.drop(columns=["ca", "thal", "chol"])

    # there are columns with missing values: "restecg", "slope", "trestbps" , and "thalch". we remove any row with missing values in these columns.
    data = data.dropna(subset=["restecg", "slope", "trestbps", "thalch"])      
     
    
    # Encode categorical variables: The categorical variables in the dataset are "sex", "cp", "fbs", "restecg", "exang", and "slope". We will use one-hot encoding for these variables.
    categorical_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


    print("Data preprocessing completed.")
    return data


def save_preprocessed_data(data: pd.DataFrame, file_path: str):
    """
    Save the preprocessed data to a CSV file.

    Parameters:
    data (pd.DataFrame): The preprocessed data as a pandas DataFrame.
    file_path (str): The path to save the CSV file.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Preprocessed data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")


if __name__ == "__main__":
    # Example usage:
    loaded_data = load_data("data/heart_disease_uci.csv")
    preprocessed_data = preprocess_data(loaded_data)
    
    if preprocessed_data is not None:
        
        print('-' * 50)
        print("Data Preview:")
        print(preprocessed_data.head())
        print('-' * 50)
        print("\nData Info:")
        print(preprocessed_data.info())
        print('-' * 50)
        print("\nData Description:")
        print(preprocessed_data.describe())
        print('-' * 50)
        save_preprocessed_data(preprocessed_data, "data/preprocessed_data.csv")

    
    else:       
        print("Failed to process data.")


