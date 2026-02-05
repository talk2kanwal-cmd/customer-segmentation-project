"""
Flask API Server for Customer Churn Prediction System
Provides REST endpoints for the web interface
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import ChurnPredictionPipeline
from data_generator import generate_customer_data
from config import (SAVED_MODELS_DIR, EVALUATION_DIR, INTERPRETATION_DIR, 
                    EDA_DIR, DATA_GENERATION)
from utils import setup_logging

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

logger = setup_logging()

# Available models
AVAILABLE_MODELS = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm']

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available trained models with their metrics"""
    try:
        # Check which models are available
        available = []
        for model_name in AVAILABLE_MODELS:
            model_path = SAVED_MODELS_DIR / f'{model_name}.pkl'
            if model_path.exists():
                available.append(model_name)
        
        # Try to load comparison metrics if available
        comparison_path = EVALUATION_DIR / 'model_comparison.csv'
        metrics = {}
        
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            for _, row in df.iterrows():
                model_name = row['Model']
                metrics[model_name] = {
                    'roc_auc': float(row['ROC-AUC']),
                    'precision': float(row['Precision']),
                    'recall': float(row['Recall']),
                    'f1_score': float(row['F1-Score'])
                }
        
        return jsonify({
            'success': True,
            'models': available,
            'metrics': metrics
        })
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-comparison', methods=['GET'])
def get_model_comparison():
    """Get model comparison data"""
    try:
        comparison_path = EVALUATION_DIR / 'model_comparison.csv'
        
        if not comparison_path.exists():
            return jsonify({
                'success': False,
                'error': 'Model comparison data not found. Please train models first.'
            }), 404
        
        df = pd.read_csv(comparison_path)
        
        return jsonify({
            'success': True,
            'data': df.to_dict(orient='records')
        })
    
    except Exception as e:
        logger.error(f"Error getting model comparison: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make churn predictions from uploaded data"""
    try:
        # Get model name from request
        model_name = request.form.get('model', 'xgboost')
        
        if model_name not in AVAILABLE_MODELS:
            return jsonify({
                'success': False,
                'error': f'Invalid model: {model_name}'
            }), 400
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({
                    'success': False,
                    'error': 'Only CSV files are supported'
                }), 400
            
            # Read CSV file
            df = pd.read_csv(file)
            
        elif request.is_json:
            # Handle JSON data
            data = request.get_json()
            df = pd.DataFrame(data)
        
        else:
            return jsonify({
                'success': False,
                'error': 'No data provided. Please upload a CSV file or send JSON data.'
            }), 400
        
        # Validate required columns
        required_cols = ['tenure_days', 'monthly_charges', 'total_charges', 'num_logins', 
                        'num_transactions', 'num_support_tickets', 'payment_failures',
                        'subscription_tier', 'age_group', 'region']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {", ".join(missing_cols)}'
            }), 400
        
        # Initialize pipeline and make predictions
        logger.info(f"Making predictions using {model_name}")
        pipeline = ChurnPredictionPipeline(model_name=model_name)
        results = pipeline.predict(df)
        
        # Calculate risk distribution
        risk_distribution = results['risk_category'].value_counts().to_dict()
        
        # Get summary statistics
        summary = {
            'total_customers': len(results),
            'high_risk': len(results[results['risk_category'] == 'High Risk']),
            'medium_risk': len(results[results['risk_category'] == 'Medium Risk']),
            'low_risk': len(results[results['risk_category'] == 'Low Risk']),
            'avg_churn_probability': float(results['churn_probability'].mean()),
            'max_churn_probability': float(results['churn_probability'].max()),
            'min_churn_probability': float(results['churn_probability'].min())
        }
        
        return jsonify({
            'success': True,
            'model': model_name,
            'summary': summary,
            'risk_distribution': risk_distribution,
            'predictions': results.to_dict(orient='records')
        })
    
    except FileNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Model not found. Please train models first by running main.py'
        }), 404
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-sample', methods=['POST'])
def generate_sample():
    """Generate sample customer data for testing"""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 100)
        churn_rate = data.get('churn_rate', 0.23)
        
        # Generate data
        logger.info(f"Generating {n_samples} sample customers")
        df = generate_customer_data(
            n_samples=n_samples,
            churn_rate=churn_rate,
            random_seed=None  # Random each time
        )
        
        # Remove the churned column for prediction
        if 'churned' in df.columns:
            df = df.drop(columns=['churned'])
        
        return jsonify({
            'success': True,
            'data': df.to_dict(orient='records'),
            'count': len(df)
        })
    
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/visualizations/<path:filename>', methods=['GET'])
def get_visualization(filename):
    """Serve visualization images"""
    try:
        # Check in different output directories
        for directory in [EVALUATION_DIR, INTERPRETATION_DIR, EDA_DIR]:
            file_path = directory / filename
            if file_path.exists():
                return send_file(file_path, mimetype='image/png')
        
        return jsonify({
            'success': False,
            'error': f'Visualization not found: {filename}'
        }), 404
    
    except Exception as e:
        logger.error(f"Error serving visualization: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'online',
        'message': 'Customer Churn Prediction API is running'
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "CUSTOMER CHURN PREDICTION WEB INTERFACE")
    print("="*70)
    print(f"\n🚀 Server starting on http://localhost:5000")
    print(f"📊 Available models: {', '.join(AVAILABLE_MODELS)}")
    print(f"📁 Static files: {Path(__file__).parent / 'static'}")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
