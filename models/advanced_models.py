"""
Advanced Models for Churn Prediction
Implements Random Forest, XGBoost, and LightGBM classifiers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_PARAMS, SAVED_MODELS_DIR
from utils import setup_logging, save_model, calculate_metrics, print_metrics

class AdvancedModels:
    """Advanced ensemble classification models"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.models = {}
        
    def train_random_forest(self, X_train, y_train, class_weight='balanced'):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weight: Class weight strategy
            
        Returns:
            Trained model
        """
        self.logger.info("Training Random Forest")
        
        params = MODEL_PARAMS['random_forest'].copy()
        params['class_weight'] = class_weight
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.logger.info("Random Forest training completed")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            
        Returns:
            Trained model
        """
        self.logger.info("Training XGBoost")
        
        params = MODEL_PARAMS['xgboost'].copy()
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        if X_val is not None and y_val is not None:
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        self.logger.info("XGBoost training completed")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            
        Returns:
            Trained model
        """
        self.logger.info("Training LightGBM")
        
        params = MODEL_PARAMS['lightgbm'].copy()
        
        # Calculate class weights
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        if X_val is not None and y_val is not None:
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
        else:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        self.logger.info("LightGBM training completed")
        
        return model
    
    def get_feature_importance(self, model, feature_names, top_n=10):
        """
        Extract feature importance from tree-based models
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        else:
            self.logger.warning("Model does not have feature_importances_ attribute")
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for display
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print metrics
        print_metrics(metrics, model_name)
        
        return metrics
    
    def train_all_advanced_models(self, X_train, y_train, X_val, y_val, 
                                  save_models=True):
        """
        Train all advanced models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of trained models and their metrics
        """
        self.logger.info("Training all advanced models")
        
        results = {}
        
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(rf_model, X_val, y_val, "Random Forest")
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics
        }
        
        # XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        xgb_metrics = self.evaluate_model(xgb_model, X_val, y_val, "XGBoost")
        results['xgboost'] = {
            'model': xgb_model,
            'metrics': xgb_metrics
        }
        
        # LightGBM
        lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_metrics = self.evaluate_model(lgb_model, X_val, y_val, "LightGBM")
        results['lightgbm'] = {
            'model': lgb_model,
            'metrics': lgb_metrics
        }
        
        # Save models
        if save_models:
            save_model(rf_model, SAVED_MODELS_DIR / 'random_forest.pkl')
            save_model(xgb_model, SAVED_MODELS_DIR / 'xgboost.pkl')
            save_model(lgb_model, SAVED_MODELS_DIR / 'lightgbm.pkl')
            self.logger.info("Saved advanced models")
        
        return results

def main():
    """Train and evaluate advanced models"""
    from preprocessing import DataPreprocessor
    
    logger = setup_logging()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline()
    
    # Train advanced models
    advanced = AdvancedModels()
    results = advanced.train_all_advanced_models(X_train, y_train, X_val, y_val)
    
    # Compare models
    print("\n" + "="*60)
    print("ADVANCED MODELS COMPARISON")
    print("="*60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        print(f"  Precision: {result['metrics']['precision']:.4f}")
        print(f"  Recall: {result['metrics']['recall']:.4f}")
        print(f"  F1-Score: {result['metrics']['f1']:.4f}")
    print("="*60)
    
    # Show feature importance for Random Forest
    print("\n" + "="*60)
    print("TOP 10 FEATURES (Random Forest)")
    print("="*60)
    feature_names = X_train.columns.tolist()
    importance_df = advanced.get_feature_importance(
        results['random_forest']['model'], 
        feature_names, 
        top_n=10
    )
    print(importance_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
