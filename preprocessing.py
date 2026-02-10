"""
Data Preprocessing Module
Handles data cleaning, transformation, and train/test splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import json

sys.path.append(str(Path(__file__).parent))
from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_PARAMS, 
                    DATA_GENERATION, RANDOM_SEED)
from utils import setup_logging, load_data, save_data, save_model
from feature_engineering import FeatureEngineer

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_raw_data(self, filepath=None):
        """Load raw customer data"""
        if filepath is None:
            filepath = DATA_GENERATION['output_file']
        
        self.logger.info(f"Loading raw data from {filepath}")
        self.df = load_data(filepath)
        return self.df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values")
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            self.logger.info(f"Missing values found:\n{missing_summary[missing_summary > 0]}")
            
            # Impute numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    self.logger.info(f"Imputed {col} with median: {median_value:.2f}")
            
            # Impute categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                    self.logger.info(f"Imputed {col} with mode: {mode_value}")
        else:
            self.logger.info("No missing values found")
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        self.logger.info("Encoding categorical features")
        
        categorical_cols = ['subscription_tier', 'age_group', 'region']
        self.categorical_columns = categorical_cols
        
        # Use one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        self.logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df_encoded
    
    def scale_numerical_features(self, df, fit=True):
        """Scale numerical features"""
        self.logger.info("Scaling numerical features")
        
        # Identify numerical columns (excluding target and ID)
        exclude_cols = ['customer_id', 'churned']
        numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                         if col not in exclude_cols]
        
        self.numerical_columns = numerical_cols
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self.logger.info(f"Fitted and transformed {len(numerical_cols)} numerical columns")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            self.logger.info(f"Transformed {len(numerical_cols)} numerical columns")
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets"""
        self.logger.info("Splitting data into train/val/test sets")
        
        # Separate features and target
        X = df.drop(['customer_id', 'churned'], axis=1)
        y = df['churned']
        customer_ids = df['customer_id']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
            X, y, customer_ids, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_temp, y_temp, ids_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        self.logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(df):.1%})")
        self.logger.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df):.1%})")
        self.logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(df):.1%})")
        
        # Check churn distribution
        self.logger.info(f"Train churn rate: {y_train.mean():.2%}")
        self.logger.info(f"Val churn rate: {y_val.mean():.2%}")
        self.logger.info(f"Test churn rate: {y_test.mean():.2%}")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                ids_train, ids_val, ids_test)
    
    def preprocess_pipeline(self, input_file=None, save_processed=True):
        """Complete preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline")
        
        # Load data
        df = self.load_raw_data(input_file)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Apply feature engineering
        engineer = FeatureEngineer()
        df = engineer.create_all_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Split data BEFORE scaling (to prevent data leakage)
        splits = self.split_data(
            df, 
            test_size=MODEL_PARAMS['test_size'],
            val_size=MODEL_PARAMS['val_size'],
            random_state=MODEL_PARAMS['random_seed']
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test = splits
        
        # Scale features (fit on train, transform on val and test)
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        if save_processed:
            # Save processed data
            train_df = X_train_scaled.copy()
            train_df['customer_id'] = ids_train.values
            train_df['churned'] = y_train.values
            save_data(train_df, PROCESSED_DATA_DIR / 'train.csv')
            
            val_df = X_val_scaled.copy()
            val_df['customer_id'] = ids_val.values
            val_df['churned'] = y_val.values
            save_data(val_df, PROCESSED_DATA_DIR / 'val.csv')
            
            test_df = X_test_scaled.copy()
            test_df['customer_id'] = ids_test.values
            test_df['churned'] = y_test.values
            save_data(test_df, PROCESSED_DATA_DIR / 'test.csv')
            
            # Save scaler
            save_model(self.scaler, PROCESSED_DATA_DIR / 'scaler.pkl')
            
            # Save feature names for prediction consistency
            with open(PROCESSED_DATA_DIR / 'feature_names.json', 'w') as f:
                json.dump(X_train.columns.tolist(), f)
            
            self.logger.info("Saved processed data, scaler, and feature names")
        
        self.logger.info("Preprocessing pipeline completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def main():
    """Run preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline()
    
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nFeature columns: {X_train.shape[1]}")
    print(f"Sample features: {list(X_train.columns[:5])}")
    print("="*60)

if __name__ == "__main__":
    main()
