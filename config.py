"""
Configuration file for Customer Churn Prediction System
Contains all paths, parameters, and settings
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Directory structure
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
INTERPRETATION_DIR = OUTPUTS_DIR / "interpretation"

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, 
                  EDA_DIR, EVALUATION_DIR, INTERPRETATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data generation parameters
DATA_GENERATION = {
    'n_samples': 10000,
    'churn_rate': 0.23,  # 23% churn rate
    'random_seed': 42,
    'output_file': RAW_DATA_DIR / 'customer_data.csv'
}

# Feature engineering parameters
FEATURE_ENGINEERING = {
    'engagement_weights': {
        'login_frequency': 0.4,
        'session_duration': 0.3,
        'feature_usage_count': 0.3
    },
    'risk_thresholds': {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.0
    }
}

# Model training parameters
MODEL_PARAMS = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_seed': 42,
    'cv_folds': 5,
    
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    },
    
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 50,
        'random_state': 42
    },
    
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 20,
        'random_state': 42,
        'n_jobs': -1
    },
    
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'auc'
    },
    
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
}

# Class imbalance handling
IMBALANCE_STRATEGY = 'smote'  # Options: 'smote', 'class_weight', 'threshold'
SMOTE_PARAMS = {
    'sampling_strategy': 'auto',
    'random_state': 42,
    'k_neighbors': 5
}

# Evaluation parameters
EVALUATION = {
    'metrics': ['roc_auc', 'precision', 'recall', 'f1', 'accuracy'],
    'threshold_range': (0.3, 0.7, 0.05),  # start, stop, step
    'business_costs': {
        'false_positive': 10,  # Cost of unnecessary retention effort
        'false_negative': 100  # Cost of losing a customer
    }
}

# Visualization settings
VIZ_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'Set2'
}

# Random seed for reproducibility
RANDOM_SEED = 42
