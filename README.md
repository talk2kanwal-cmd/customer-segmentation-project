# Customer Churn Prediction System

A comprehensive machine learning system for predicting customer churn and engagement risk in subscription-based digital platforms. This project implements supervised classification models with advanced feature engineering, class imbalance handling, and interpretable insights for business decision-making.

## 🎯 Project Overview

This system analyzes historical customer behavioral data to identify users likely to disengage, enabling proactive retention strategies through:
- **Predictive Analytics**: Binary classification (churned vs. retained)
- **Risk Segmentation**: Multi-level risk categorization (high, medium, low)
- **Actionable Insights**: SHAP-based model interpretation for business teams

## 📋 Features

### Core ML Features
- **Synthetic Data Generation**: Realistic customer behavioral dataset with 10,000+ samples
- **Comprehensive EDA**: Statistical analysis and visualizations
- **Advanced Feature Engineering**: Engagement scores, recency metrics, interaction features
- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- **Class Imbalance Handling**: SMOTE, class weights, threshold optimization
- **Robust Evaluation**: ROC-AUC, Precision-Recall curves, confusion matrices
- **Model Interpretation**: SHAP values, feature importance, business insights
- **Production Pipeline**: End-to-end prediction pipeline for new customers

### 🌐 Web Interface (NEW!)
- **Modern UI**: Dark theme with glassmorphism effects and vibrant gradients
- **Drag & Drop Upload**: Easy CSV file upload with visual feedback
- **Sample Data Generation**: Instantly generate test data for quick demos
- **Model Selection**: Choose from 5 trained models with live performance metrics
- **Real-time Predictions**: Get churn risk analysis in seconds
- **Interactive Visualizations**: Beautiful charts with Chart.js
- **Risk Categorization**: Automatic high/medium/low risk segmentation
- **Download Results**: Export predictions as CSV files

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone or navigate to the project directory**
```bash
cd "f:\work\uet lahore\first semester\projects\customer segmentation project"
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Usage

### 🌐 Option 1: Web Interface (Recommended)

The easiest way to use the system is through the modern web interface:

```bash
# Start the web server
python web/app.py
```

Then open your browser to `http://localhost:5000`

**Features:**
- 📤 Drag and drop CSV files or generate sample data
- 🤖 Select from 5 trained ML models
- 📊 View interactive visualizations and risk analysis
- 💾 Download prediction results as CSV

See [web/README.md](web/README.md) for detailed web interface documentation.

### 🖥️ Option 2: Command Line - Full Pipeline

Run the complete end-to-end workflow:
```bash
python main.py --mode full
```

This executes:
1. Data generation
2. Exploratory data analysis
3. Preprocessing and feature engineering
4. Model training (5 algorithms)
5. Model evaluation and comparison
6. SHAP-based interpretation

### Individual Components

**Generate Data Only**
```bash
python data_generator.py
```

**Run EDA**
```bash
python exploratory_analysis.py
```

**Train Models**
```bash
python main.py --mode train
```

**Make Predictions (Python API)**
```python
from pipeline import ChurnPredictionPipeline

# Initialize pipeline with best model
pipeline = ChurnPredictionPipeline(model_name='xgboost')

# Predict from DataFrame
predictions = pipeline.predict(customer_df)

# Predict from CSV file
predictions = pipeline.predict_from_file('new_customers.csv', 'predictions.csv')
```

## 📁 Project Structure

```
customer-segmentation-project/
├── data/
│   ├── raw/                    # Raw customer data
│   └── processed/              # Preprocessed train/val/test splits
├── models/
│   ├── saved/                  # Trained model files
│   ├── baseline_models.py      # Logistic Regression, Decision Tree
│   ├── advanced_models.py      # Random Forest, XGBoost, LightGBM
│   └── imbalance_handler.py    # Class imbalance techniques
├── outputs/
│   ├── eda/                    # EDA visualizations
│   ├── evaluation/             # Model performance plots
│   └── interpretation/         # SHAP plots, feature importance
├── web/                        # 🌐 Web Interface
│   ├── app.py                  # Flask API server
│   ├── static/                 # Frontend files
│   │   ├── index.html          # Main page
│   │   ├── css/styles.css      # Dark theme + glassmorphism
│   │   └── js/app.js           # Interactive functionality
│   └── README.md               # Web interface documentation
├── config.py                   # Configuration and parameters
├── utils.py                    # Utility functions
├── data_generator.py           # Synthetic data generation
├── exploratory_analysis.py     # EDA module
├── preprocessing.py            # Data preprocessing
├── feature_engineering.py      # Feature creation
├── evaluation.py               # Model evaluation
├── model_interpretation.py     # SHAP analysis
├── pipeline.py                 # Prediction pipeline
├── main.py                     # Main orchestration script
└── requirements.txt            # Python dependencies
```

## 🔬 Model Performance

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Decision Tree | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| XGBoost | **0.XXX** | **0.XXX** | **0.XXX** | **0.XXX** |
| LightGBM | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

*Note: Run the pipeline to populate actual metrics*

## 💡 Key Insights

Top churn risk indicators (based on feature importance):
1. **Payment Failures**: Strong predictor of churn
2. **Days Since Last Login**: Inactivity correlates with churn
3. **Engagement Score**: Low engagement indicates high risk
4. **Support Tickets**: High volume suggests dissatisfaction
5. **Tenure**: Newer customers have higher churn risk

## 🎯 Business Recommendations

Based on model insights:
- **High-Risk Customers**: Immediate intervention with personalized offers
- **Payment Issues**: Proactive payment support and flexible billing
- **Inactive Users**: Re-engagement campaigns (email, push notifications)
- **Support Escalation**: Fast-track resolution for high-ticket customers
- **Onboarding Focus**: Enhanced support for customers in first 3 months

## 🔧 Configuration

Edit `config.py` to customize:
- Dataset size and churn rate
- Model hyperparameters
- Feature engineering weights
- Risk thresholds
- Evaluation metrics

## 📈 Outputs

After running the pipeline, find:
- **EDA Visualizations**: `outputs/eda/`
- **Model Performance**: `outputs/evaluation/`
- **SHAP Analysis**: `outputs/interpretation/`
- **Trained Models**: `models/saved/`
- **Processed Data**: `data/processed/`

## 🧪 Testing

Verify installation and functionality:
```bash
# Test data generation
python data_generator.py

# Test preprocessing
python preprocessing.py

# Test individual model
python models/baseline_models.py
```

## 📝 Requirements

### Core ML Dependencies
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- shap >= 0.42.0
- imbalanced-learn >= 0.11.0

### Web Interface Dependencies
- flask >= 3.0.0
- flask-cors >= 4.0.0

See `requirements.txt` for complete list.

## 🤝 Contributing

This is an academic project for UET Lahore. For questions or suggestions, please contact the project maintainer.

## 📄 License

This project is created for educational purposes as part of the First Semester coursework at UET Lahore.

## 🙏 Acknowledgments

- UET Lahore - First Semester Projects
- scikit-learn, XGBoost, and LightGBM communities
- SHAP library for model interpretability

---

## 🌟 Highlights

### 🎨 Modern Web Interface
Beautiful, production-ready interface with:
- Dark theme with purple-blue-pink gradients
- Glassmorphism effects and smooth animations
- Real-time predictions and interactive charts
- Drag-and-drop file upload
- Mobile-responsive design

### 🤖 5 Machine Learning Models
- Logistic Regression (baseline)
- Decision Tree (interpretable)
- Random Forest (ensemble)
- XGBoost (best performance)
- LightGBM (fast training)

### 📊 Comprehensive Analysis
- SHAP-based feature importance
- Risk categorization (High/Medium/Low)
- Interactive visualizations
- Exportable results

---

**Project Status**: ✅ Complete with Web Interface

**Last Updated**: February 2026
