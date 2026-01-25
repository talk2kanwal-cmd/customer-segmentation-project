"""
Model Evaluation Module
Comprehensive evaluation with multiple metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             confusion_matrix, classification_report)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import EVALUATION_DIR, EVALUATION
from utils import setup_logging, calculate_metrics

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.results = {}
        
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model", 
                      output_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for title
            output_path: Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, 
                                    model_name="Model", output_path=None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for title
            output_path: Path to save figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", 
                             output_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name for title
            output_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', 
                 fontsize=14, fontweight='bold')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def compare_models(self, models_results, output_path=None):
        """
        Compare multiple models
        
        Args:
            models_results: Dictionary of {model_name: {'metrics': {...}}}
            output_path: Path to save comparison plot
        """
        # Extract metrics for comparison
        comparison_data = []
        for model_name, result in models_results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': metrics['roc_auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.2
        
        metrics_to_plot = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i * width, df[metric], width, label=metric, color=colors[i])
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return df
    
    def evaluate_with_threshold(self, y_true, y_pred_proba, threshold=0.5):
        """
        Evaluate model with custom threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        metrics['threshold'] = threshold
        
        return metrics
    
    def find_optimal_threshold(self, y_true, y_pred_proba, 
                              cost_fp=10, cost_fn=100):
        """
        Find optimal threshold based on business costs
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
            
        Returns:
            Optimal threshold and associated metrics
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            metrics = self.evaluate_with_threshold(y_true, y_pred_proba, threshold)
            
            # Calculate business cost
            cost = (metrics['false_positives'] * cost_fp + 
                   metrics['false_negatives'] * cost_fn)
            
            results.append({
                'threshold': threshold,
                'cost': cost,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
        
        results_df = pd.DataFrame(results)
        optimal_idx = results_df['cost'].idxmin()
        optimal_result = results_df.iloc[optimal_idx]
        
        self.logger.info(f"Optimal threshold: {optimal_result['threshold']:.2f}")
        self.logger.info(f"Minimum cost: {optimal_result['cost']:.2f}")
        
        return optimal_result
    
    def generate_classification_report(self, y_true, y_pred, output_path=None):
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save report
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=['Retained', 'Churned'])
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report

def main():
    """Demonstrate evaluation capabilities"""
    # Create sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.77, 0.23])
    y_pred_proba = np.random.beta(2, 5, size=n_samples)
    y_pred_proba[y_true == 1] += 0.3
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Evaluate
    evaluator = ModelEvaluator()
    
    # Plot ROC curve
    evaluator.plot_roc_curve(y_true, y_pred_proba, "Sample Model",
                            EVALUATION_DIR / 'sample_roc_curve.png')
    
    # Plot PR curve
    evaluator.plot_precision_recall_curve(y_true, y_pred_proba, "Sample Model",
                                         EVALUATION_DIR / 'sample_pr_curve.png')
    
    # Find optimal threshold
    optimal = evaluator.find_optimal_threshold(y_true, y_pred_proba)
    print(f"\nOptimal Threshold: {optimal['threshold']:.2f}")
    print(f"Precision: {optimal['precision']:.3f}")
    print(f"Recall: {optimal['recall']:.3f}")
    print(f"F1-Score: {optimal['f1']:.3f}")

if __name__ == "__main__":
    main()
