🚀 Employee ROI Prediction & Explainability System
📌 Overview

This project is an AI-powered Employee ROI Prediction System that combines:

📊 Machine Learning (XGBoost) for ROI prediction
🔍 SHAP Explainability for feature-level insights
🧠 LLM (OpenAI) for human-like business analysis

It helps HR teams understand:

Why an employee’s ROI is high/low
Key performance drivers
Risks like burnout or inefficiency
Actionable recommendations
🧠 System Architecture
Dataset → Feature Engineering → XGBoost Model → ROI Prediction
                                              ↓
                                        SHAP Analysis
                                              ↓
                                Rule-based Insights + Recommendations
                                              ↓
                                  OpenAI LLM Enhancement
📂 Project Structure
project/
│── dataset.csv           # Training dataset
│── model.pkl             # Trained model
│── model.py              # Model training script
│── main.py               # Prediction + SHAP + LLM pipeline
│── .env                  # API key (not committed)
│── README.md
⚙️ Features
✅ 1. ROI Prediction
Uses XGBoost Regressor
Optimized with:
Learning rate
Depth
Subsampling
Includes monotonic constraints for realistic behavior
✅ 2. Feature Engineering

Two key engineered features:

Efficiency

tasks_completed / (working_hours + 1)

Work Pressure

deadlines_missed / (tasks_completed + 1)
✅ 3. Explainability (SHAP)
Uses TreeExplainer
Shows feature contribution to ROI
Post-processed for business logic:
Salary → always negative impact
Deadlines missed → negative
Tasks completed → positive
✅ 4. Rule-Based Insights

Generates insights like:

Productivity issues
Skill gaps
Workload imbalance
✅ 5. Smart Recommendations

Examples:

Reduce working hours → burnout risk
Improve efficiency → training
Salary vs performance mismatch
✅ 6. LLM Enhancement (OpenAI)

Final layer adds:

Deep reasoning
Risk analysis
Business-level explanation
🛠️ Installation
1. Clone Repo
git clone https://github.com/your-username/roi-prediction.git
cd roi-prediction
2. Install Dependencies
pip install pandas numpy scikit-learn xgboost shap openai python-dotenv joblib
🔐 Environment Setup

Create a .env file:

OPENAI_API_KEY=your_api_key_here
▶️ How to Run
🔹 Train Model
python model.py

This will:

Train XGBoost model
Evaluate performance
Save model as model.pkl
🔹 Run Prediction + Analysis
python main.py

Output includes:

Predicted ROI
SHAP feature contributions
Insights
Recommendations
LLM enhanced analysis
📊 Model Details
Algorithm:
XGBRegressor
Metrics:
MAE (Mean Absolute Error)
RMSE
R² Score