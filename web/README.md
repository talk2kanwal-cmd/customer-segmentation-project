# Customer Churn Prediction Web Interface

A modern, interactive web application for the Customer Churn Prediction System.

## 🚀 Features

- **Drag & Drop Upload**: Easy CSV file upload with visual feedback
- **Sample Data Generation**: Generate test data instantly
- **Model Selection**: Choose from 5 trained ML models
- **Real-time Predictions**: Get churn risk analysis instantly
- **Interactive Visualizations**: Beautiful charts with Chart.js
- **Risk Segmentation**: Automatic high/medium/low risk categorization
- **Downloadable Results**: Export predictions as CSV

## 📋 Prerequisites

- Python 3.8+
- All models trained (run `main.py --mode full` from project root)
- Flask dependencies installed

## ⚙️ Installation

1. **Install web interface dependencies**:
   ```bash
   pip install flask flask-cors
   ```

2. **Verify models are trained**:
   ```bash
   # Check if models exist
   dir models\saved
   ```

   If models don't exist, train them first:
   ```bash
   python main.py --mode full
   ```

## 🌐 Running the Web Interface

1. **Navigate to the web directory**:
   ```bash
   cd web
   ```

2. **Start the Flask server**:
   ```bash
   python app.py
   ```

3. **Open your browser**:
   Navigate to `http://localhost:5000`

The server will start on port 5000 and serve the web interface.

## 🎯 Usage

### 1. Upload Customer Data

**Option A: Upload CSV File**
- Click the upload area or drag & drop a CSV file
- Required columns:
  - `customer_id`
  - `tenure_days`
  - `monthly_charges`
  - `total_charges`
  - `num_logins`
  - `num_transactions`
  - `num_support_tickets`
  - `payment_failures`
  - `subscription_tier`
  - `age_group`
  - `region`

**Option B: Generate Sample Data**
- Click "Generate Sample Data" button
- 100 sample customers will be created automatically

### 2. Select a Model

Choose from the available trained models:
- **Logistic Regression**: Fast, interpretable baseline
- **Decision Tree**: Simple rule-based classifier
- **Random Forest**: Ensemble method, robust
- **XGBoost**: Gradient boosting, high accuracy
- **LightGBM**: Fast gradient boosting

Each model card shows its performance metrics (ROC-AUC, Precision, Recall, F1-Score).

### 3. Run Prediction

Click the **"Predict Churn Risk"** button to:
- Process customer data through the pipeline
- Apply feature engineering
- Generate churn predictions
- Categorize risk levels

### 4. View Results

The results dashboard displays:
- **Summary Cards**: High/Medium/Low risk counts
- **Risk Distribution Chart**: Visual breakdown
- **Predictions Table**: Customer-level predictions
- **Download Button**: Export full results as CSV

## 📊 API Endpoints

### GET `/api/models`
Get list of available trained models with metrics.

**Response**:
```json
{
  "success": true,
  "models": ["logistic_regression", "xgboost", ...],
  "metrics": {
    "xgboost": {
      "roc_auc": 0.99,
      "precision": 0.98,
      "recall": 0.97,
      "f1_score": 0.98
    }
  }
}
```

### POST `/api/predict`
Make churn predictions.

**Request**:
- **Form Data**:
  - `file`: CSV file
  - `model`: Model name (e.g., "xgboost")

**Response**:
```json
{
  "success": true,
  "model": "xgboost",
  "summary": {
    "total_customers": 100,
    "high_risk": 23,
    "medium_risk": 35,
    "low_risk": 42,
    "avg_churn_probability": 0.34
  },
  "predictions": [...]
}
```

### POST `/api/generate-sample`
Generate sample customer data.

**Request**:
```json
{
  "n_samples": 100,
  "churn_rate": 0.23
}
```

### GET `/api/model-comparison`
Get model comparison metrics.

### GET `/api/health`
Health check endpoint.

## 🎨 Design Features

### Modern Dark Theme
- Deep navy background with gradient accents
- Purple, blue, and pink color scheme
- High contrast for readability

### Glassmorphism Effects
- Semi-transparent cards with backdrop blur
- Subtle borders and shadows
- Premium, modern aesthetic

### Smooth Animations
- Hover effects on all interactive elements
- Floating upload icon
- Fade-in animations for sections
- Pulse animations for stats

### Responsive Design
- Works on desktop, tablet, and mobile
- Flexible grid layouts
- Adaptive font sizes

## 🔧 Troubleshooting

### Models Not Found
**Error**: "Model not found. Please train models first"

**Solution**:
```bash
# Navigate to project root
cd ..

# Train all models
python main.py --mode full
```

### Port Already in Use
**Error**: "Address already in use"

**Solution**:
```bash
# Find process using port 5000 (Windows)
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F

# Or change port in app.py
app.run(port=5001)
```

### CSV Upload Fails
**Error**: "Missing required columns"

**Solution**:
- Ensure CSV has all required columns
- Use "Generate Sample Data" to see correct format
- Check for typos in column names

## 📁 Project Structure

```
web/
├── app.py                  # Flask API server
├── static/
│   ├── index.html          # Main HTML page
│   ├── css/
│   │   └── styles.css      # Comprehensive styles
│   └── js/
│       └── app.js          # Frontend JavaScript
└── README.md               # This file
```

## 🌟 Tips

1. **Best Model**: XGBoost and Random Forest typically perform best
2. **Sample Data**: Use sample data to explore the interface without real data
3. **Download Results**: Always download CSV for full prediction dataset
4. **Risk Thresholds**: Configured in `config.py` (High: >70%, Medium: 40-70%, Low: <40%)

## 🚀 Next Steps

- Deploy to cloud (AWS, Azure, GCP)
- Add user authentication
- Implement prediction history
- Add batch processing for large files
- Create API key management

---

**Built with**: Flask, HTML5, CSS3, JavaScript, Chart.js
**Author**: UET Lahore - First Semester Project
**Last Updated**: February 2026
