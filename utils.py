"""
Utility functions for the Customer Churn Prediction System
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib

def setup_logging(log_file: str = 'churn_prediction.log', level=logging.INFO):
    """
    Configure logging for the project
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df

def save_data(df: pd.DataFrame, filepath: Path):
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
    """
    logger = logging.getLogger(__name__)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved data to {filepath}")

def save_model(model: Any, filepath: Path):
    """
    Save trained model using joblib
    
    Args:
        model: Trained model object
        filepath: Output file path
    """
    logger = logging.getLogger(__name__)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Saved model to {filepath}")

def load_model(filepath: Path) -> Any:
    """
    Load trained model using joblib
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model object
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)

def plot_distribution(data: pd.Series, title: str, xlabel: str, 
                     output_path: Path = None, bins: int = 30):
    """
    Plot distribution of a variable
    
    Args:
        data: Series to plot
        title: Plot title
        xlabel: X-axis label
        output_path: Optional path to save figure
        bins: Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, title: str, 
                            output_path: Path = None, figsize: Tuple = (12, 10)):
    """
    Plot correlation heatmap
    
    Args:
        df: DataFrame with numerical columns
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    correlation = df.corr()
    
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, confusion_matrix)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    return metrics

def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        if metric not in ['true_negatives', 'false_positives', 
                         'false_negatives', 'true_positives']:
            print(f"{metric.upper():.<30} {value:.4f}")
    
    if 'true_positives' in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Positives:  {metrics['true_positives']}")
    
    print(f"{'='*50}\n")

def create_directory_structure(base_dir: Path):
    """
    Create project directory structure
    
    Args:
        base_dir: Base project directory
    """
    directories = [
        'data/raw',
        'data/processed',
        'models/saved',
        'outputs/eda',
        'outputs/evaluation',
        'outputs/interpretation'
    ]
    
    for dir_path in directories:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    logging.info("Created directory structure")
