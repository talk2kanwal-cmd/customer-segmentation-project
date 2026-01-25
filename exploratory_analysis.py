"""
Exploratory Data Analysis Module
Comprehensive EDA with visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import EDA_DIR, DATA_GENERATION
from utils import setup_logging, load_data

class ExploratoryAnalyzer:
    """Performs comprehensive EDA on customer data"""
    
    def __init__(self):
        self.logger = setup_logging()
        sns.set_style("whitegrid")
        
    def load_and_summarize(self, filepath):
        """Load data and print summary statistics"""
        self.logger.info(f"Loading data from {filepath}")
        self.df = load_data(filepath)
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Shape: {self.df.shape}")
        print(f"\nColumn Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        print("="*60)
        
        return self.df
    
    def analyze_churn_distribution(self, output_path=None):
        """Analyze and visualize churn distribution"""
        self.logger.info("Analyzing churn distribution")
        
        churn_counts = self.df['churned'].value_counts()
        churn_pct = self.df['churned'].value_counts(normalize=True) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        axes[0].bar(['Retained', 'Churned'], churn_counts.values, 
                   color=['#2ecc71', '#e74c3c'])
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Churn Distribution (Count)', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # Percentage plot
        axes[1].pie(churn_pct.values, labels=['Retained', 'Churned'], 
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                   startangle=90)
        axes[1].set_title('Churn Distribution (%)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_numerical_distributions(self, output_path=None):
        """Plot distributions of numerical features by churn status"""
        self.logger.info("Plotting numerical feature distributions")
        
        numerical_cols = ['tenure_months', 'login_frequency', 'session_duration',
                         'feature_usage_count', 'days_since_last_login',
                         'support_tickets', 'payment_failures']
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            if col in self.df.columns:
                self.df[self.df['churned'] == 0][col].hist(
                    ax=axes[idx], bins=30, alpha=0.6, label='Retained', color='green'
                )
                self.df[self.df['churned'] == 1][col].hist(
                    ax=axes[idx], bins=30, alpha=0.6, label='Churned', color='red'
                )
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                axes[idx].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_categorical_distributions(self, output_path=None):
        """Plot distributions of categorical features by churn status"""
        self.logger.info("Plotting categorical feature distributions")
        
        categorical_cols = ['subscription_tier', 'age_group', 'region']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, col in enumerate(categorical_cols):
            if col in self.df.columns:
                churn_by_cat = pd.crosstab(self.df[col], self.df['churned'], normalize='index') * 100
                churn_by_cat.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'])
                axes[idx].set_xlabel(col, fontsize=11)
                axes[idx].set_ylabel('Percentage (%)', fontsize=11)
                axes[idx].set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
                axes[idx].legend(['Retained', 'Churned'])
                axes[idx].grid(axis='y', alpha=0.3)
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_heatmap(self, output_path=None):
        """Plot correlation heatmap of numerical features"""
        self.logger.info("Creating correlation heatmap")
        
        numerical_df = self.df.select_dtypes(include=[np.number])
        correlation = numerical_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def statistical_tests(self):
        """Perform statistical tests for feature significance"""
        self.logger.info("Performing statistical tests")
        
        results = []
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'churned']
        
        for col in numerical_cols:
            churned = self.df[self.df['churned'] == 1][col].dropna()
            retained = self.df[self.df['churned'] == 0][col].dropna()
            
            # T-test
            t_stat, p_value = stats.ttest_ind(churned, retained)
            
            results.append({
                'feature': col,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': 'Yes' if p_value < 0.05 else 'No'
            })
        
        results_df = pd.DataFrame(results).sort_values('p_value')
        
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTS (T-Test)")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)
        
        return results_df
    
    def run_full_eda(self, filepath=None):
        """Run complete EDA pipeline"""
        if filepath is None:
            filepath = DATA_GENERATION['output_file']
        
        self.logger.info("Starting full EDA pipeline")
        
        # Load and summarize
        self.load_and_summarize(filepath)
        
        # Churn distribution
        self.analyze_churn_distribution(EDA_DIR / 'churn_distribution.png')
        
        # Numerical distributions
        self.plot_numerical_distributions(EDA_DIR / 'numerical_distributions.png')
        
        # Categorical distributions
        self.plot_categorical_distributions(EDA_DIR / 'categorical_distributions.png')
        
        # Correlation heatmap
        self.plot_correlation_heatmap(EDA_DIR / 'correlation_heatmap.png')
        
        # Statistical tests
        self.statistical_tests()
        
        self.logger.info(f"EDA completed. Visualizations saved to {EDA_DIR}")
        
        print(f"\n[OK] EDA completed successfully!")
        print(f"[OK] Visualizations saved to: {EDA_DIR}")

def main():
    """Run EDA"""
    analyzer = ExploratoryAnalyzer()
    analyzer.run_full_eda()

if __name__ == "__main__":
    main()
