🚀 Monthly Employee ROI Prediction & AI Analytics System
📌 Overview

This project is an advanced Monthly Employee ROI Prediction System that combines:

📊 Machine Learning (XGBoost) for ROI prediction
🔍 SHAP Explainability for feature-level contribution
🧠 OpenAI LLM for intelligent HR insights

It helps organizations analyze monthly employee performance and answer:

Why ROI is high or low
Which factors are driving performance
What risks exist (burnout, inefficiency, pressure)
What actions should HR take
🧠 System Architecture
Monthly Dataset → Feature Engineering → XGBoost Model → ROI Prediction
                                                     ↓
                                               SHAP Analysis
                                                     ↓
                                  Rule-Based Insights & Recommendations
                                                     ↓
                                     OpenAI LLM Enhancement Layer
📂 Project Structure
project/
│── dataset.csv   # Training dataset
│── model.pkl                        # Saved trained model
│── model.py                         # Model training script
│── main.py                          # Prediction + SHAP + LLM pipeline
│── .env                             # OpenAI API key
│── README.md
⚙️ Key Features
✅ 1. Monthly ROI Prediction
Uses XGBoost Regressor
Handles real-world HR patterns using:
Regularization (L1, L2)
Subsampling
Monotonic constraints
✅ 2. Advanced Feature Engineering
📊 Core Monthly Features:
monthly_tasks_completed
monthly_working_hours
monthly_deadlines_missed
monthly_active_days
⚡ Engineered Features:

1. Monthly Efficiency

monthly_tasks_completed / (monthly_working_hours + 1)

2. Monthly Work Pressure

(monthly_deadlines_missed * 2) + (monthly_working_hours / 200)

3. Monthly Productivity

monthly_tasks_completed / (monthly_active_days + 1)
✅ 3. SHAP Explainability
Uses TreeExplainer
Explains:
Feature contribution to ROI
Positive vs negative impact
Post-processing ensures:
Deadlines → always negative
Tasks → always positive
✅ 4. Rule-Based Insights Engine

Automatically detects:

Efficiency issues
Skill gaps
Overwork patterns
Performance mismatches
✅ 5. Smart Recommendations System

Examples:

Reduce working hours → burnout prevention
Improve efficiency → training
Rebalance workload → reduce pressure
Salary-performance alignment
✅ 6. LLM (OpenAI) Enhancement Layer

Final output is enhanced with:

Business-level reasoning
Risk analysis
Strategic recommendations
🛠️ Installation

 Install Dependencies
pip install pandas numpy scikit-learn xgboost shap openai python-dotenv joblib
🔐 Environment Setup

Create a .env file:

OPENAI_API_KEY=your_api_key_here

⚠️ Add .env to .gitignore

▶️ Usage
🔹 Train Model
python model.py

✔️ Output:

Model performance (MAE, RMSE, R²)
Feature importance
Saved model → monthly_roi_model.pkl
🔹 Run Prediction & Analysis
python main.py

✔️ Output:

Predicted ROI
SHAP feature contributions
Insights
Recommendations
LLM enhanced analysis
📊 Model Details
Algorithm:
XGBRegressor
Optimization:
Learning rate: 0.05
Estimators: 500
Depth: 8
Regularization: L1 + L2
📈 Evaluation Metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score
📌 Monotonic Constraints Logic
Feature	Impact
Experience	Positive
Job Role Level	Positive
Salary	Controlled / Cost
Working Hours	Slight negative
Deadlines Missed	Negative
Tasks Completed	Positive
Efficiency	Positive
Work Pressure	Negative
Skill Score	Positive