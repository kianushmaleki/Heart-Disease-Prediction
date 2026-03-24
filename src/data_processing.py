import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
    



def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values and encoding categorical variables.

    Parameters:
    data (pd.DataFrame): The raw data as a pandas DataFrame.

    Returns:
    pd.DataFrame: The cleaned data.
    """
    #drop the id column and dataset
    data = data.drop(columns=["id", "dataset"])

    # Handle missing values: There are three columns with most missing values: "ca", "thal", and "chol". We do not use these columns in our model, so we drop them.
    data = data.drop(columns=["ca", "thal", "chol"])

    # there are columns with missing values: "restecg", "slope", "trestbps" , and "thalch". we remove any row with missing values in these columns.
    data = data.dropna(subset=["restecg", "slope", "trestbps", "thalch"])      
     
    
    # Encode categorical variables: The categorical variables in the dataset are "sex", "cp", "fbs", "restecg", "exang", and "slope". We will use one-hot encoding for these variables.
    categorical_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # use StandardScaler to scale the numerical features: "age", "trestbps", "thalch", and "oldpeak".
    numerical_columns = ["age", "trestbps", "thalch", "oldpeak"]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


    
    print("Data cleaning completed.")
    return data


def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the cleaned data to a CSV file.

    Parameters:
    data (pd.DataFrame): The cleaned data as a pandas DataFrame.
    file_path (str): The path to save the CSV file.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Cleaned data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")


def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Parameters:
    data (pd.DataFrame): The cleaned data as a pandas DataFrame.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    tuple of pd.DataFrame: X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print("Data splitting completed.")
    save_data(X_train, "data/X_train.csv")
    save_data(X_test, "data/X_test.csv")
    save_data(y_train, "data/y_train.csv")
    save_data(y_test, "data/y_test.csv")   
    print("Train and test sets saved successfully.")
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    # Example usage:
    loaded_data = load_data("data/heart_disease_uci.csv")
    cleaned_data = clean_data(loaded_data)
    
    if cleaned_data is not None:
        
        print('-' * 50)
        print("Data Preview:")
        print(cleaned_data.head())
        print('-' * 50)
        print("\nData Info:")
        print(cleaned_data.info())
        print('-' * 50)
        print("\nData Description:")
        print(cleaned_data.describe())
        print('-' * 50)
        split_data(cleaned_data, target_column="num")
        print('-' * 50)

    else:       
        print("Failed to process data.")


