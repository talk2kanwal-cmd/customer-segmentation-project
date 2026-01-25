"""
Main Orchestration Script
Runs the complete churn prediction workflow
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging
from data_generator import generate_customer_data
from exploratory_analysis import ExploratoryAnalyzer
from preprocessing import DataPreprocessor
from models.baseline_models import BaselineModels
from models.advanced_models import AdvancedModels
from evaluation import ModelEvaluator
from model_interpretation import ModelInterpreter
from config import (DATA_GENERATION, EVALUATION_DIR, INTERPRETATION_DIR, 
                    SAVED_MODELS_DIR)

def run_data_generation():
    """Generate synthetic customer data"""
    logger = setup_logging()
    logger.info("="*60)
    logger.info("STEP 1: DATA GENERATION")
    logger.info("="*60)
    
    df = generate_customer_data(
        n_samples=DATA_GENERATION['n_samples'],
        churn_rate=DATA_GENERATION['churn_rate'],
        random_seed=DATA_GENERATION['random_seed']
    )
    
    print(f"\n[OK] Generated {len(df)} customer records")
    print(f"[OK] Churn rate: {df['churned'].mean():.2%}")
    
    return df

def run_eda():
    """Run exploratory data analysis"""
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)
    
    analyzer = ExploratoryAnalyzer()
    analyzer.run_full_eda()
    
    print(f"\n[OK] EDA completed")

def run_preprocessing():
    """Run data preprocessing"""
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline()
    
    print(f"\n[OK] Preprocessing completed")
    print(f"[OK] Train set: {X_train.shape}")
    print(f"[OK] Validation set: {X_val.shape}")
    print(f"[OK] Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_model_training(X_train, y_train, X_val, y_val):
    """Train all models"""
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*60)
    
    all_results = {}
    
    # Train baseline models
    print("\nTraining baseline models...")
    baseline = BaselineModels()
    baseline_results = baseline.train_all_baseline_models(X_train, y_train, X_val, y_val)
    all_results.update(baseline_results)
    
    # Train advanced models
    print("\nTraining advanced models...")
    advanced = AdvancedModels()
    advanced_results = advanced.train_all_advanced_models(X_train, y_train, X_val, y_val)
    all_results.update(advanced_results)
    
    print(f"\n[OK] Trained {len(all_results)} models")
    
    return all_results

def run_evaluation(all_results, X_test, y_test):
    """Evaluate all models"""
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("="*60)
    
    evaluator = ModelEvaluator()
    
    # Evaluate each model on test set
    test_results = {}
    for model_name, result in all_results.items():
        model = result['model']
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Plot ROC curve
        evaluator.plot_roc_curve(
            y_test, y_pred_proba, model_name,
            EVALUATION_DIR / f'{model_name}_roc_curve.png'
        )
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name,
            EVALUATION_DIR / f'{model_name}_confusion_matrix.png'
        )
        
        # Calculate metrics
        from utils import calculate_metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        test_results[model_name] = {'metrics': metrics}
    
    # Compare all models
    comparison_df = evaluator.compare_models(
        test_results,
        EVALUATION_DIR / 'model_comparison.png'
    )
    
    # Save comparison
    comparison_df.to_csv(EVALUATION_DIR / 'model_comparison.csv', index=False)
    
    print(f"\n[OK] Model evaluation completed")
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
    best_roc_auc = comparison_df['ROC-AUC'].max()
    
    print(f"\n[BEST] Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    return test_results, best_model_name

def run_interpretation(all_results, best_model_name, X_test):
    """Run model interpretation"""
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("STEP 6: MODEL INTERPRETATION")
    logger.info("="*60)
    
    interpreter = ModelInterpreter()
    best_model = all_results[best_model_name]['model']
    
    # Feature importance
    print(f"\nAnalyzing {best_model_name} feature importance...")
    importance_df = interpreter.plot_feature_importance(
        best_model,
        X_test.columns.tolist(),
        top_n=15,
        output_path=INTERPRETATION_DIR / 'feature_importance.png'
    )
    
    # SHAP analysis
    print(f"Calculating SHAP values...")
    
    # Check model type for SHAP
    if 'logistic' in best_model_name.lower():
        model_type = 'linear'
    else:
        model_type = 'tree'
        
    shap_values, X_sample = interpreter.calculate_shap_values(
        best_model, X_test, model_type=model_type
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
    insights = interpreter.generate_insights(importance_df, top_n=10)
    
    print(f"\n[OK] Model interpretation completed")
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    for insight in insights:
        print(insight)
    print("="*60)
    
    return insights

def run_full_pipeline():
    """Run complete end-to-end pipeline"""
    logger = setup_logging()
    
    print("\n" + "="*70)
    print(" "*15 + "CUSTOMER CHURN PREDICTION SYSTEM")
    print("="*70)
    
    # Step 1: Data Generation
    run_data_generation()
    
    # Step 2: EDA
    run_eda()
    
    # Step 3: Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing()
    
    # Step 4: Model Training
    all_results = run_model_training(X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluation
    test_results, best_model_name = run_evaluation(all_results, X_test, y_test)
    
    # Step 6: Interpretation
    run_interpretation(all_results, best_model_name, X_test)
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n[INFO] Visualizations saved to: outputs/")
    print(f"[INFO] Models saved to: {SAVED_MODELS_DIR}")
    print(f"[BEST] Best model: {best_model_name}")
    print("\n" + "="*70)

def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(description='Customer Churn Prediction System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'generate', 'eda', 'preprocess', 'train', 'evaluate'],
                       help='Pipeline mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'generate':
        run_data_generation()
    elif args.mode == 'eda':
        run_eda()
    elif args.mode == 'preprocess':
        run_preprocessing()
    elif args.mode == 'train':
        X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing()
        run_model_training(X_train, y_train, X_val, y_val)
    elif args.mode == 'evaluate':
        X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing()
        all_results = run_model_training(X_train, y_train, X_val, y_val)
        run_evaluation(all_results, X_test, y_test)

if __name__ == "__main__":
    main()
