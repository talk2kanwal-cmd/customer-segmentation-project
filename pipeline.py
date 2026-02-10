"""
End-to-End Prediction Pipeline
Handles data loading, preprocessing, and prediction for new customers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent))
from config import SAVED_MODELS_DIR, PROCESSED_DATA_DIR, FEATURE_ENGINEERING
from utils import setup_logging, load_model, load_data, save_data
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor

class ChurnPredictionPipeline:
    """End-to-end pipeline for churn prediction"""
    
    def __init__(self, model_name='xgboost'):
        """
        Initialize pipeline
        
        Args:
            model_name: Name of the model to use for predictions
        """
        self.logger = setup_logging()
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.risk_thresholds = FEATURE_ENGINEERING['risk_thresholds']
        self.feature_names = None
        
    def load_model_and_scaler(self):
        """Load trained model and scaler"""
        self.logger.info(f"Loading {self.model_name} model")
        
        model_path = SAVED_MODELS_DIR / f'{self.model_name}.pkl'
        scaler_path = PROCESSED_DATA_DIR / 'scaler.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = load_model(model_path)
        
        if scaler_path.exists():
            self.scaler = load_model(scaler_path)
            self.logger.info("Loaded scaler")
        else:
            self.logger.warning("Scaler not found, predictions may be inaccurate")
        
        # Load feature names
        feature_names_path = PROCESSED_DATA_DIR / 'feature_names.json'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            self.logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            self.logger.warning("Feature names JSON not found, using raw columns")
        
        return self.model
    
    def preprocess_input(self, df):
        """
        Preprocess input data for prediction
        
        Args:
            df: DataFrame with raw customer data
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        self.logger.info("Preprocessing input data")
        
        # Handle missing values
        df = self.preprocessor.handle_missing_values(df)
        
        # Apply feature engineering
        df = self.feature_engineer.create_all_features(df)
        
        # Encode categorical features
        categorical_cols = ['subscription_tier', 'age_group', 'region']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Scale features
        # Ensure we have all columns in the correct order
        if self.feature_names is not None:
            # Reindex to match training columns, filling missing dummies with 0
            X = df_encoded.reindex(columns=self.feature_names, fill_value=0)
            self.logger.info("Aligned input features with training set")
        else:
            feature_cols = [col for col in df_encoded.columns 
                           if col not in ['customer_id', 'churned']]
            X = df_encoded[feature_cols]
        
        # Scale features
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return X_scaled
    
    def predict(self, df):
        """
        Make churn predictions
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            self.load_model_and_scaler()
        
        self.logger.info(f"Making predictions for {len(df)} customers")
        
        # Store customer IDs
        customer_ids = df['customer_id'].values if 'customer_id' in df.columns else None
        
        # Preprocess
        X = self.preprocess_input(df)
        
        # Predict using values to avoid feature name mismatch warnings/errors
        predictions = self.model.predict(X.values)
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X.values)[:, 1]
        else:
            # For models that don't support predict_proba, use decision_function or fallback
            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X.values)
                probabilities = 1 / (1 + np.exp(-scores))  # Sigmoid
            else:
                probabilities = predictions.astype(float)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': customer_ids if customer_ids is not None else range(len(df)),
            'churn_probability': probabilities,
            'predicted_churn': predictions,
            'risk_category': self.categorize_risk(probabilities)
        })
        
        self.logger.info("Predictions completed")
        
        return results
    
    def categorize_risk(self, probabilities):
        """
        Categorize customers into risk levels
        
        Args:
            probabilities: Array of churn probabilities
            
        Returns:
            Array of risk categories
        """
        risk_categories = []
        
        for prob in probabilities:
            if prob >= self.risk_thresholds['high']:
                risk_categories.append('High Risk')
            elif prob >= self.risk_thresholds['medium']:
                risk_categories.append('Medium Risk')
            else:
                risk_categories.append('Low Risk')
        
        return risk_categories
    
    def predict_from_file(self, input_path, output_path=None):
        """
        Make predictions from CSV file
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Loading data from {input_path}")
        df = load_data(Path(input_path))
        
        # Make predictions
        results = self.predict(df)
        
        # Save if output path provided
        if output_path:
            save_data(results, Path(output_path))
            self.logger.info(f"Predictions saved to {output_path}")
        
        return results
    
    def batch_predict(self, df, batch_size=1000):
        """
        Make predictions in batches for large datasets
        
        Args:
            df: DataFrame with customer data
            batch_size: Number of samples per batch
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Batch prediction for {len(df)} customers")
        
        results_list = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = self.predict(batch)
            results_list.append(batch_results)
        
        results = pd.concat(results_list, ignore_index=True)
        
        return results

def main():
    """Demonstrate pipeline usage"""
    from data_generator import generate_customer_data
    from config import DATA_GENERATION
    
    logger = setup_logging()
    
    # Generate sample data for prediction
    logger.info("Generating sample customer data")
    sample_df = generate_customer_data(
        n_samples=100,
        churn_rate=0.23,
        random_seed=123
    )
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline(model_name='xgboost')
    
    try:
        # Make predictions
        results = pipeline.predict(sample_df)
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS SAMPLE")
        print("="*60)
        print(results.head(10))
        print("\n" + "="*60)
        print("RISK DISTRIBUTION")
        print("="*60)
        print(results['risk_category'].value_counts())
        print("="*60)
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.info("Please train models first by running main.py")

if __name__ == "__main__":
    main()
