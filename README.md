# Heart Disease Prediction: End-to-End MLOps Pipeline

## Project Overview
This project implements a complete, production-ready MLOps pipeline for predicting heart disease. Rather than focusing solely on model accuracy, this project demonstrates the infrastructure and automation required to manage a machine learning lifecycle, applying best practices in version control, experiment tracking, automated testing, CI/CD, and drift monitoring.

The pipeline includes:
*   **Version Control**: Git for code and DVC for data tracking.
*   **Experiment Tracking**: MLflow for logging hyperparameters, metrics, and model artifacts.
*   **Automated Testing**: `pytest` suite covering unit tests, data validation, and model performance.
*   **CI/CD**: GitHub Actions workflows to automate testing and training.
*   **Monitoring**: Drift detection using Evidently to identify feature distribution shifts.

## Dataset
The project uses the **Heart Disease Prediction** dataset from the UCI Machine Learning Repository. 
- **Task**: Binary Classification (Predicting presence of heart disease).
- **Features**: 14 attributes including age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Preprocessing**: The pipeline handles missing values (imputation) and encodes categorical variables.

## Repository Structure
```text
├── .github/workflows/
│   └── pipeline.yml        # GitHub Actions CI/CD definition
├── configs/
│   └── config.yaml         # Model hyperparameters and file paths
├── data/                   # Data directory (tracked by DVC)
│   └── heart.csv.dvc       # DVC pointer file
├── reports/                # Generated drift reports
├── src/                    # Source code
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── train.py            # Model training and MLflow logging
│   ├── evaluate.py         # Model evaluation metrics
│   ├── compare_runs.py     # Programmatic MLflow query script
│   └── monitor_drift.py    # Drift detection script using Evidently
├── tests/                  # Pytest suite
│   ├── test_preprocess.py  # Unit tests for data logic
│   ├── test_data.py        # Data validation tests
│   └── test_model.py       # Model validation tests
├── requirements.txt        # Pinned dependencies
├── .gitignore              # Excludes data, models, and caches
└── README.md
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd heart-disease-prediction
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Retrieval (DVC)
The dataset is versioned using DVC. To pull the data from the remote storage:
```bash
dvc pull
```

## Workflow

### Experiment Tracking (MLflow)
To run the training pipeline and log results to MLflow:
```bash
python src/train.py
```
All hyperparameters are loaded from `configs/config.yaml`. To identify the best performing model across multiple runs:
```bash
python src/compare_runs.py
```

### Running Tests
The test suite validates preprocessing, data integrity, and model behavior:
```bash
pytest tests/ -v
```

### Drift Monitoring
To check for data drift between the training set and "production" data:
```bash
python src/monitor_drift.py
```
This generates an HTML report in the `reports/` directory and exits with a non-zero code if drift exceeds the threshold.

---

## Drift Analysis & Monitoring Report

### 1. Which features showed drift and why?
In the simulated production environment, the features **Cholesterol (chol)** and **Oldpeak** showed significant statistical drift.
*   **Analysis**: This drift was observed when comparing the baseline training set against simulated incoming data. The shift in cholesterol levels suggests a change in the patient demographic or a change in the data collection source.

### 2. Would this drift likely affect model performance?
**Yes.** Heart disease prediction relies heavily on these physiological markers. 
- **Cholesterol** is a primary indicator; if the model was trained on a population with a lower mean, it may under-predict risk for the current population.
- **Oldpeak** (ST depression induced by exercise) is a sensitive indicator of heart stress. A significant shift here directly impacts the model's sensitivity and specificity.

### 3. Recommendations
Based on the Evidently report results:
*   **Action**: **Investigate and Retrain.**
*   **Justification**: Because key predictive features showed drift exceeding our threshold, the model should be retrained. I recommend checking if this is a "data bug" (e.g., unit change) or a "real-world shift." If it is a real-world shift, the new data must be incorporated into the training pipeline to maintain accuracy.

---

## CI/CD Pipeline
The GitHub Actions workflow (`.github/workflows/pipeline.yml`) ensures high quality through two primary stages:
1.  **Test Job**: Installs dependencies and runs the full `pytest` suite.
2.  **Train Job**: If tests pass, the model is trained. The job verifies that the model meets minimum performance thresholds (e.g., Accuracy > 0.80) before completing successfully.


