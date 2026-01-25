"""
Baseline Models for Churn Prediction
Implements Logistic Regression and Decision Tree classifiers
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_PARAMS, SAVED_MODELS_DIR
from utils import setup_logging, save_model, calculate_metrics, print_metrics

class BaselineModels:
    """Baseline classification models"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.models = {}
        
    def train_logistic_regression(self, X_train, y_train, class_weight='balanced'):
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weight: Class weight strategy
            
        Returns:
            Trained model
        """
        self.logger.info("Training Logistic Regression")
        
        params = MODEL_PARAMS['logistic_regression'].copy()
        params['class_weight'] = class_weight
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        self.logger.info("Logistic Regression training completed")
        
        return model
    
    def train_decision_tree(self, X_train, y_train, class_weight='balanced'):
        """
        Train Decision Tree model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weight: Class weight strategy
            
        Returns:
            Trained model
        """
        self.logger.info("Training Decision Tree")
        
        params = MODEL_PARAMS['decision_tree'].copy()
        params['class_weight'] = class_weight
        
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['decision_tree'] = model
        self.logger.info("Decision Tree training completed")
        
        return model
    
    def cross_validate(self, model, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        self.logger.info(f"Performing {cv}-fold cross-validation")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, 
                             random_state=MODEL_PARAMS['random_seed'])
        
        scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        self.logger.info(f"CV ROC-AUC scores: {scores}")
        self.logger.info(f"Mean CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
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
    
    def train_all_baseline_models(self, X_train, y_train, X_val, y_val, 
                                  save_models=True):
        """
        Train all baseline models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of trained models and their metrics
        """
        self.logger.info("Training all baseline models")
        
        results = {}
        
        # Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_metrics = self.evaluate_model(lr_model, X_val, y_val, "Logistic Regression")
        results['logistic_regression'] = {
            'model': lr_model,
            'metrics': lr_metrics
        }
        
        # Decision Tree
        dt_model = self.train_decision_tree(X_train, y_train)
        dt_metrics = self.evaluate_model(dt_model, X_val, y_val, "Decision Tree")
        results['decision_tree'] = {
            'model': dt_model,
            'metrics': dt_metrics
        }
        
        # Save models
        if save_models:
            save_model(lr_model, SAVED_MODELS_DIR / 'logistic_regression.pkl')
            save_model(dt_model, SAVED_MODELS_DIR / 'decision_tree.pkl')
            self.logger.info("Saved baseline models")
        
        return results

def main():
    """Train and evaluate baseline models"""
    from preprocessing import DataPreprocessor
    
    logger = setup_logging()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline()
    
    # Train baseline models
    baseline = BaselineModels()
    results = baseline.train_all_baseline_models(X_train, y_train, X_val, y_val)
    
    # Compare models
    print("\n" + "="*60)
    print("BASELINE MODELS COMPARISON")
    print("="*60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        print(f"  Precision: {result['metrics']['precision']:.4f}")
        print(f"  Recall: {result['metrics']['recall']:.4f}")
        print(f"  F1-Score: {result['metrics']['f1']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
