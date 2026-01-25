"""
Model Interpretation Module
SHAP-based model interpretation and feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import INTERPRETATION_DIR
from utils import setup_logging

class ModelInterpreter:
    """Model interpretation using SHAP and feature importance"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.explainer = None
        self.shap_values = None
        
    def plot_feature_importance(self, model, feature_names, top_n=15, 
                               output_path=None):
        """
        Plot feature importance from tree-based or linear models
        
        Args:
            model: Trained model with feature_importances_ or coef_
            feature_names: List of feature names
            top_n: Number of top features to display
            output_path: Path to save figure
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Use absolute values of coefficients for linear models
            importance = np.abs(model.coef_[0])
        else:
            self.logger.warning("Model does not have feature_importances_ or coef_ attribute")
            return None
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Return importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def calculate_shap_values(self, model, X_data, model_type='tree'):
        """
        Calculate SHAP values for model interpretation
        
        Args:
            model: Trained model
            X_data: Data to explain (sample for efficiency)
            model_type: 'tree' for tree-based models, 'linear' for linear models
            
        Returns:
            SHAP values
        """
        self.logger.info(f"Calculating SHAP values for {model_type} model")
        
        # Sample data if too large
        if len(X_data) > 500:
            self.logger.info(f"Sampling {500} instances for SHAP calculation")
            X_sample = X_data.sample(n=500, random_state=42)
        else:
            X_sample = X_data
        
        # Create appropriate explainer
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_sample)
        else:
            self.explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)
        
        # For binary classification, get values for positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        self.logger.info("SHAP values calculated successfully")
        
        return self.shap_values, X_sample
    
    def plot_shap_summary(self, shap_values, X_data, output_path=None):
        """
        Plot SHAP summary plot
        
        Args:
            shap_values: SHAP values
            X_data: Data used for SHAP calculation
            output_path: Path to save figure
        """
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_data, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Churn Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_shap_bar(self, shap_values, X_data, output_path=None):
        """
        Plot SHAP bar plot (mean absolute SHAP values)
        
        Args:
            shap_values: SHAP values
            X_data: Data used for SHAP calculation
            output_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Mean |SHAP value|)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_shap_waterfall(self, shap_values, X_data, instance_idx=0, 
                           output_path=None):
        """
        Plot SHAP waterfall plot for a single prediction
        
        Args:
            shap_values: SHAP values
            X_data: Data used for SHAP calculation
            instance_idx: Index of instance to explain
            output_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        # Create explanation object for waterfall plot
        if hasattr(shap, 'Explanation'):
            explanation = shap.Explanation(
                values=shap_values[instance_idx],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X_data.iloc[instance_idx].values,
                feature_names=X_data.columns.tolist()
            )
            shap.waterfall_plot(explanation, show=False)
        else:
            # Fallback for older SHAP versions
            shap.force_plot(
                self.explainer.expected_value,
                shap_values[instance_idx],
                X_data.iloc[instance_idx],
                matplotlib=True,
                show=False
            )
        
        plt.title(f'SHAP Explanation for Instance {instance_idx}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_insights(self, importance_df, top_n=10):
        """
        Generate business insights from feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to analyze
            
        Returns:
            List of insight strings
        """
        insights = []
        top_features = importance_df.head(top_n)
        
        insights.append(f"Top {top_n} Churn Risk Indicators:")
        
        for idx, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Generate human-readable insights
            if 'payment_failure' in feature.lower():
                insights.append(
                    f"• Payment failures are a critical churn indicator "
                    f"(importance: {importance:.3f})"
                )
            elif 'days_since_last_login' in feature.lower():
                insights.append(
                    f"• Customer inactivity (days since last login) strongly predicts churn "
                    f"(importance: {importance:.3f})"
                )
            elif 'engagement' in feature.lower():
                insights.append(
                    f"• Low engagement scores indicate high churn risk "
                    f"(importance: {importance:.3f})"
                )
            elif 'support_ticket' in feature.lower():
                insights.append(
                    f"• High support ticket volume correlates with churn "
                    f"(importance: {importance:.3f})"
                )
            elif 'tenure' in feature.lower():
                insights.append(
                    f"• Customer tenure affects churn probability "
                    f"(importance: {importance:.3f})"
                )
            else:
                insights.append(
                    f"• {feature} is a significant churn predictor "
                    f"(importance: {importance:.3f})"
                )
        
        return insights

def main():
    """Demonstrate interpretation capabilities"""
    from models.advanced_models import AdvancedModels
    from preprocessing import DataPreprocessor
    
    logger = setup_logging()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline()
    
    # Train a model
    logger.info("Training Random Forest for interpretation")
    advanced = AdvancedModels()
    rf_model = advanced.train_random_forest(X_train, y_train)
    
    # Interpret model
    interpreter = ModelInterpreter()
    
    # Feature importance
    importance_df = interpreter.plot_feature_importance(
        rf_model, 
        X_train.columns.tolist(),
        top_n=15,
        output_path=INTERPRETATION_DIR / 'feature_importance.png'
    )
    
    # SHAP analysis
    shap_values, X_sample = interpreter.calculate_shap_values(
        rf_model, X_test, model_type='tree'
    )
    
    interpreter.plot_shap_summary(
        shap_values, X_sample,
        output_path=INTERPRETATION_DIR / 'shap_summary.png'
    )
    
    interpreter.plot_shap_bar(
        shap_values, X_sample,
        output_path=INTERPRETATION_DIR / 'shap_bar.png'
    )
    
    # Generate insights
    insights = interpreter.generate_insights(importance_df)
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    for insight in insights:
        print(insight)
    print("="*60)

if __name__ == "__main__":
    main()
