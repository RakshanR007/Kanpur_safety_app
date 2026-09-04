# 🚸 Kanpur Child Pedestrian Safety Profile Classifier

An end-to-end Machine Learning web application predicting child pedestrian crossing safety profiles (Safe vs. Risky) using survey data from Kanpur, India.

## 📊 Model Performance Benchmarks
- **Architecture**: Random Forest Classifier with SMOTE Balancing & Chi-Squared Feature Selection
- **Holdout Test Accuracy**: 90.48%
- **5-Fold Cross-Validation Accuracy**: 94.29%
- **Macro Recall / Sensitivity**: 88.00% / 92.00%
- **ROC-AUC**: 98.24%

## 🚀 Deployment
This application is configured for direct deployment on **Streamlit Community Cloud**.

### Key Files:
- `app.py`: Interactive Streamlit dashboard
- `student_risk_pipeline.pkl`: Serialized production ML pipeline
- `model_metadata.json`: Feature specifications and performance benchmarks
- `requirements.txt`: Python package dependencies
