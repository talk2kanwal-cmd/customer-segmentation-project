# Customer Churn Prediction System

A production-grade machine learning system for predicting customer churn and engagement risk in subscription-based digital platforms. This project implements advanced supervised classification models with sophisticated feature engineering, robustness to missing data, and interpretable business insights.

## 🎯 Project Overview

This system analyzes historical customer behavioral data to identify users likely to disengage, enabling proactive retention strategies through:
- **Predictive Analytics**: Binary classification (churned vs. retained) with 5 trained models.
- **Robust Feature Engineering**: 29+ engineered features capturing engagement, recency, and interactions.
- **Modern Web Interface**: Glassmorphic, dark-themed dashboard for real-time predictions and data visualization.
- **Production-Ready Pipeline**: End-to-end handling of missing values, feature synchronization, and model serving.

## 📋 Features

### Core ML & Architecture
- **Multi-Model Support**: Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.
- **Advanced Feature Engineering**: engagement scores, recency ratios, interaction terms, and behavioral change indicators.
- **Data Reliability**: Robust handling of `NaN` values and missing categorical data using median/mode imputation.
- **Feature Consistency**: Automated synchronization of feature order between training and prediction via `feature_names.json`.
- **Model Interpretation**: SHAP-based analysis and automated feature importance visualization.

### 🌐 Modern Web Dashboard
- **Real-time Analytics**: Instant risk scores and categorization (High/Medium/Low Risk).
- **Interactive Charts**: Dynamic performance metrics and risk distribution via Chart.js.
- **Drag & Drop Upload**: Support for CSV file analysis with live feedback.
- **Sample Generation**: One-click test data generation to explore the model's capabilities.
- **Exportable Results**: Download prediction reports as CSV files.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Option 1: Web Interface (Recommended)
Start the modern dashboard to interact with the models:
```bash
python web/app.py
```
Visit `http://localhost:5000` in your browser.

### Option 2: Command Line (Full Pipeline)
Run the entire workflow from data generation to interpretation:
```bash
python main.py --mode full
```

### Option 3: Individual Stages
- **Generate Data**: `python data_generator.py`
- **Train Models**: `python main.py --mode train`
- **Evaluation**: `python main.py --mode evaluate`

## 📁 Project Structure

```text
customer-segmentation-project/
├── data/
│   ├── raw/                    # Raw customer datasets
│   └── processed/              # Splits, Scaler, and feature_names.json
├── models/
│   ├── saved/                  # Trained model binaries (.pkl)
│   ├── baseline_models.py      # Logistic Regression, Decision Tree
│   └── advanced_models.py      # Random Forest, XGBoost, LightGBM
├── web/                        # 🌐 Flask Web Interface
│   ├── app.py                  # API Server
│   └── static/                 # Frontend (HTML, CSS, JS)
├── main.py                     # Pipeline Orchestrator
├── preprocessing.py            # Robust data cleaning & imputation
├── feature_engineering.py      # Advanced signal creation
└── pipeline.py                 # Production prediction engine
```

## 🔬 Reliability & Consistency
The system implements several specific mechanisms to ensure production reliability:
- **NaN Defense**: Missing behavioral data is automatically imputed with training-time medians.
- **JSON Integrity**: APIs are hardened against invalid math literals (sending `null` instead of `NaN`).
- **Feature Alignment**: The `pipeline.py` strictly reindexes input data to match the model's expected feature order, preventing common "unseen feature" errors.

## 📄 License
This project is created for educational purposes as part of the First Semester coursework at UET Lahore.

---
**Project Status**: ✅ Stable & Complete
**Last Updated**: February 2026
