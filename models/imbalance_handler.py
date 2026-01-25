"""
Class Imbalance Handler
Implements techniques to handle imbalanced churn data
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import SMOTE_PARAMS, RANDOM_SEED
from utils import setup_logging

class ImbalanceHandler:
    """Handles class imbalance in churn prediction"""
    
    def __init__(self, strategy='smote'):
        """
        Initialize imbalance handler
        
        Args:
            strategy: 'smote', 'class_weight', or 'threshold'
        """
        self.logger = setup_logging()
        self.strategy = strategy
        self.smote = None
        self.class_weights = None
        
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to balance the training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Balanced X_train, y_train
        """
        self.logger.info("Applying SMOTE for class balancing")
        
        # Check class distribution before SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        self.logger.info(f"Before SMOTE - Class distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        self.smote = SMOTE(**SMOTE_PARAMS)
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        # Check class distribution after SMOTE
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.logger.info(f"After SMOTE - Class distribution: {dict(zip(unique, counts))}")
        
        return X_resampled, y_resampled
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights for imbalanced data
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        self.logger.info("Computing class weights")
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.class_weights = dict(zip(classes, weights))
        
        self.logger.info(f"Class weights: {self.class_weights}")
        
        return self.class_weights
    
    def optimize_threshold(self, y_true, y_pred_proba, cost_fp=10, cost_fn=100):
        """
        Find optimal classification threshold based on business costs
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            cost_fp: Cost of false positive (unnecessary retention effort)
            cost_fn: Cost of false negative (losing a customer)
            
        Returns:
            Optimal threshold
        """
        self.logger.info("Optimizing classification threshold")
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            # Calculate total cost
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            costs.append(total_cost)
        
        # Find threshold with minimum cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        
        self.logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
        self.logger.info(f"Minimum cost: {costs[optimal_idx]:.2f}")
        
        return optimal_threshold
    
    def handle_imbalance(self, X_train, y_train, X_val=None, y_val=None):
        """
        Apply selected imbalance handling strategy
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for threshold optimization)
            y_val: Validation labels (for threshold optimization)
            
        Returns:
            Processed data or class weights depending on strategy
        """
        if self.strategy == 'smote':
            return self.apply_smote(X_train, y_train)
        
        elif self.strategy == 'class_weight':
            class_weights = self.compute_class_weights(y_train)
            return X_train, y_train, class_weights
        
        elif self.strategy == 'threshold':
            # This strategy doesn't modify training data
            # Threshold optimization happens during evaluation
            self.logger.info("Using threshold optimization strategy")
            return X_train, y_train
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

def compare_strategies(X_train, y_train):
    """
    Compare different imbalance handling strategies
    
    Args:
        X_train: Training features
        y_train: Training labels
    """
    logger = setup_logging()
    
    print("\n" + "="*60)
    print("IMBALANCE HANDLING STRATEGIES COMPARISON")
    print("="*60)
    
    # Original distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nOriginal class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # SMOTE
    print(f"\n1. SMOTE Strategy:")
    handler_smote = ImbalanceHandler(strategy='smote')
    X_smote, y_smote = handler_smote.apply_smote(X_train, y_train)
    unique, counts = np.unique(y_smote, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_smote)*100:.1f}%)")
    
    # Class weights
    print(f"\n2. Class Weight Strategy:")
    handler_weights = ImbalanceHandler(strategy='class_weight')
    class_weights = handler_weights.compute_class_weights(y_train)
    for cls, weight in class_weights.items():
        print(f"  Class {cls} weight: {weight:.2f}")
    
    # Threshold optimization
    print(f"\n3. Threshold Optimization Strategy:")
    print(f"  Optimizes decision threshold during evaluation")
    print(f"  No modification to training data")
    
    print("="*60)

def main():
    """Demonstrate imbalance handling"""
    # Create sample imbalanced data
    np.random.seed(RANDOM_SEED)
    n_samples = 1000
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.77, 0.23])
    
    compare_strategies(X_train, y_train)

if __name__ == "__main__":
    main()
