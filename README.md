
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
