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

if __name__ == "__main__":
    # Example usage:
    loaded_data = load_data("data/heart_disease_uci.csv")
    if loaded_data is not None:
        print('-' * 50)
        print("Data Preview:")
        print(loaded_data.head())
        print('-' * 50)
        print("\nData Info:")
        print(loaded_data.info())
        print('-' * 50)
        print("\nData Description:")
        print(loaded_data.describe())
    else:       
        print("Failed to load data.")