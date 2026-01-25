"""
Synthetic Customer Data Generator for Churn Prediction
Generates realistic customer behavioral data with churn labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import DATA_GENERATION, RAW_DATA_DIR
from utils import setup_logging, save_data

def generate_customer_data(n_samples=10000, churn_rate=0.23, random_seed=42):
    """
    Generate synthetic customer behavioral dataset
    
    Args:
        n_samples: Number of customers to generate
        churn_rate: Target churn rate (proportion of churned customers)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with customer data
    """
    np.random.seed(random_seed)
    logger = setup_logging()
    logger.info(f"Generating {n_samples} customer records with {churn_rate:.1%} churn rate")
    
    # Customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    
    # Customer attributes
    tenure_months = np.random.exponential(scale=12, size=n_samples).clip(1, 60).astype(int)
    subscription_tier = np.random.choice(['Basic', 'Standard', 'Premium'], 
                                        size=n_samples, 
                                        p=[0.5, 0.35, 0.15])
    age_group = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], 
                                 size=n_samples,
                                 p=[0.15, 0.30, 0.25, 0.20, 0.10])
    region = np.random.choice(['North', 'South', 'East', 'West'], 
                             size=n_samples,
                             p=[0.25, 0.25, 0.25, 0.25])
    
    # Behavioral metrics (influenced by churn probability)
    # Base engagement levels
    login_frequency = np.random.gamma(shape=2, scale=5, size=n_samples).clip(0, 30)
    session_duration = np.random.gamma(shape=3, scale=10, size=n_samples).clip(1, 120)
    feature_usage_count = np.random.poisson(lam=8, size=n_samples).clip(0, 50)
    
    # Recency metrics
    days_since_last_login = np.random.exponential(scale=3, size=n_samples).clip(0, 30)
    
    # Support and payment metrics
    support_tickets = np.random.poisson(lam=1.5, size=n_samples).clip(0, 20)
    payment_failures = np.random.binomial(n=5, p=0.1, size=n_samples)
    
    # Engagement indicators
    discount_usage = np.random.binomial(n=1, p=0.3, size=n_samples)
    referrals_made = np.random.poisson(lam=0.5, size=n_samples).clip(0, 10)
    
    # Premium tier users have better engagement
    premium_mask = subscription_tier == 'Premium'
    login_frequency[premium_mask] *= 1.5
    session_duration[premium_mask] *= 1.3
    feature_usage_count[premium_mask] = (feature_usage_count[premium_mask] * 1.4).astype(int)
    
    # Calculate churn probability based on behavioral signals
    churn_score = (
        -0.3 * (login_frequency / 30) +  # Low login frequency increases churn
        -0.2 * (session_duration / 120) +  # Low session duration increases churn
        -0.2 * (feature_usage_count / 50) +  # Low feature usage increases churn
        0.4 * (days_since_last_login / 30) +  # High recency increases churn
        0.3 * (support_tickets / 20) +  # Many support tickets increases churn
        0.5 * (payment_failures / 5) +  # Payment failures strongly indicate churn
        -0.1 * referrals_made / 10  # Referrals reduce churn
    )
    
    # Normalize churn score to probability
    churn_probability = 1 / (1 + np.exp(-churn_score))
    
    # Generate churn labels to match target churn rate
    churn_threshold = np.percentile(churn_probability, (1 - churn_rate) * 100)
    churned = (churn_probability >= churn_threshold).astype(int)
    
    # Adjust behavioral metrics for churned customers (make them worse)
    churned_mask = churned == 1
    login_frequency[churned_mask] *= 0.5
    session_duration[churned_mask] *= 0.6
    feature_usage_count[churned_mask] = (feature_usage_count[churned_mask] * 0.5).astype(int)
    days_since_last_login[churned_mask] *= 2.0
    support_tickets[churned_mask] = (support_tickets[churned_mask] * 1.5).astype(int)
    payment_failures[churned_mask] = (payment_failures[churned_mask] * 2).clip(0, 5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure_months': tenure_months,
        'subscription_tier': subscription_tier,
        'age_group': age_group,
        'region': region,
        'login_frequency': login_frequency.round(1),
        'session_duration': session_duration.round(1),
        'feature_usage_count': feature_usage_count,
        'days_since_last_login': days_since_last_login.round(1),
        'support_tickets': support_tickets,
        'payment_failures': payment_failures,
        'discount_usage': discount_usage,
        'referrals_made': referrals_made,
        'churned': churned
    })
    
    # Add some missing values (realistic scenario)
    missing_rate = 0.02
    for col in ['login_frequency', 'session_duration', 'support_tickets']:
        missing_mask = np.random.random(n_samples) < missing_rate
        df.loc[missing_mask, col] = np.nan
    
    logger.info(f"Generated dataset shape: {df.shape}")
    logger.info(f"Actual churn rate: {df['churned'].mean():.2%}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def main():
    """Generate and save customer data"""
    logger = setup_logging()
    
    # Generate data
    df = generate_customer_data(
        n_samples=DATA_GENERATION['n_samples'],
        churn_rate=DATA_GENERATION['churn_rate'],
        random_seed=DATA_GENERATION['random_seed']
    )
    
    # Save to CSV
    output_file = DATA_GENERATION['output_file']
    save_data(df, output_file)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CUSTOMER DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Total customers: {len(df):,}")
    print(f"Churned customers: {df['churned'].sum():,} ({df['churned'].mean():.2%})")
    print(f"Retained customers: {(1-df['churned']).sum():,} ({(1-df['churned'].mean()):.2%})")
    print(f"\nSubscription Tier Distribution:")
    print(df['subscription_tier'].value_counts())
    print(f"\nData saved to: {output_file}")
    print("="*60)
    
    logger.info("Data generation completed successfully")

if __name__ == "__main__":
    main()
