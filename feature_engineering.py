"""
Feature Engineering Module
Creates advanced features from raw customer data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import FEATURE_ENGINEERING, PROCESSED_DATA_DIR
from utils import setup_logging, load_data, save_data

class FeatureEngineer:
    """Creates advanced features for churn prediction"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.engagement_weights = FEATURE_ENGINEERING['engagement_weights']
    
    def create_engagement_score(self, df):
        """
        Create composite engagement score from multiple behavioral metrics
        
        Args:
            df: DataFrame with behavioral features
            
        Returns:
            DataFrame with engagement_score column
        """
        self.logger.info("Creating engagement score")
        
        # Normalize features to 0-1 scale
        login_norm = df['login_frequency'] / df['login_frequency'].max()
        session_norm = df['session_duration'] / df['session_duration'].max()
        usage_norm = df['feature_usage_count'] / df['feature_usage_count'].max()
        
        # Weighted combination
        df['engagement_score'] = (
            self.engagement_weights['login_frequency'] * login_norm +
            self.engagement_weights['session_duration'] * session_norm +
            self.engagement_weights['feature_usage_count'] * usage_norm
        )
        
        return df
    
    def create_recency_features(self, df):
        """Create recency-based features"""
        self.logger.info("Creating recency features")
        
        # Recency ratio (days since last login / tenure)
        df['recency_ratio'] = df['days_since_last_login'] / (df['tenure_months'] * 30 + 1)
        
        # Activity status (binary: active in last 7 days)
        df['is_recently_active'] = (df['days_since_last_login'] <= 7).astype(int)
        
        return df
    
    def create_frequency_features(self, df):
        """Create frequency-based features"""
        self.logger.info("Creating frequency features")
        
        # Support tickets per month
        df['support_tickets_per_month'] = df['support_tickets'] / (df['tenure_months'] + 1)
        
        # Payment failures per month
        df['payment_failures_per_month'] = df['payment_failures'] / (df['tenure_months'] + 1)
        
        # Login frequency per tenure
        df['login_frequency_per_tenure'] = df['login_frequency'] / (df['tenure_months'] + 1)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between variables"""
        self.logger.info("Creating interaction features")
        
        # Engagement × Tenure (loyal engaged customers)
        df['engagement_tenure_interaction'] = df['engagement_score'] * df['tenure_months']
        
        # Login frequency × Session duration (total engagement time proxy)
        df['total_engagement_proxy'] = df['login_frequency'] * df['session_duration']
        
        # Support tickets × Payment failures (problematic customer indicator)
        df['problem_indicator'] = df['support_tickets'] * (df['payment_failures'] + 1)
        
        return df
    
    def create_behavioral_change_features(self, df):
        """Create features indicating behavioral changes"""
        self.logger.info("Creating behavioral change features")
        
        # Declining usage indicator (high recency with low engagement)
        df['declining_usage'] = (
            (df['days_since_last_login'] > df['days_since_last_login'].median()) & 
            (df['engagement_score'] < df['engagement_score'].median())
        ).astype(int)
        
        # At-risk indicator (multiple negative signals)
        df['at_risk_indicator'] = (
            (df['payment_failures'] > 0) & 
            (df['support_tickets'] > df['support_tickets'].median()) &
            (df['days_since_last_login'] > 7)
        ).astype(int)
        
        return df
    
    def create_all_features(self, df):
        """
        Apply all feature engineering transformations
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering pipeline")
        
        df = df.copy()
        
        # Create all feature groups
        df = self.create_engagement_score(df)
        df = self.create_recency_features(df)
        df = self.create_frequency_features(df)
        df = self.create_interaction_features(df)
        df = self.create_behavioral_change_features(df)
        
        self.logger.info(f"Feature engineering completed. Total features: {df.shape[1]}")
        
        return df

def apply_feature_engineering_to_processed_data():
    """Apply feature engineering to already processed train/val/test sets"""
    logger = setup_logging()
    engineer = FeatureEngineer()
    
    # Load processed data
    train_df = load_data(PROCESSED_DATA_DIR / 'train.csv')
    val_df = load_data(PROCESSED_DATA_DIR / 'val.csv')
    test_df = load_data(PROCESSED_DATA_DIR / 'test.csv')
    
    logger.info("Loaded processed data splits")
    
    # Note: We need to load the RAW data and apply feature engineering BEFORE preprocessing
    # This is a better approach - let's create a combined pipeline
    logger.info("Feature engineering should be applied to raw data before preprocessing")
    logger.info("This function is for demonstration - use the combined pipeline instead")
    
    return train_df, val_df, test_df

def main():
    """Demonstrate feature engineering"""
    from data_generator import generate_customer_data
    from config import DATA_GENERATION
    
    logger = setup_logging()
    
    # Generate sample data
    df = generate_customer_data(
        n_samples=1000,
        churn_rate=DATA_GENERATION['churn_rate'],
        random_seed=DATA_GENERATION['random_seed']
    )
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    df_engineered = engineer.create_all_features(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {df_engineered.shape[1]}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")
    print(f"\nNew feature columns:")
    new_cols = set(df_engineered.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
    print("="*60)

if __name__ == "__main__":
    main()
