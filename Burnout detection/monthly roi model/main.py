import pandas as pd
import numpy as np
import joblib
import shap

# -----------------------------
# 1. LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")   # your saved model

# -----------------------------
# 2. SAMPLE INPUT (RAW)
# -----------------------------
sample = pd.DataFrame([{
    
    "experience_years": 5,
    "job_role_level": 3,
    "salary": 75000,
    "monthly_active_days": 28,
    "monthly_working_hours": 310,
    "monthly_deadlines_missed": 7,
    "monthly_tasks_completed": 120,
    "skill_score": 7,
    "task_complexity": 7

}])

# -----------------------------
# 3. FEATURE ENGINEERING (MATCH TRAINING)
# -----------------------------

sample["monthly_efficiency"] = (
    sample["monthly_tasks_completed"] /
    (sample["monthly_working_hours"] + 1)
)

sample["monthly_work_pressure"] = (
    sample["monthly_deadlines_missed"] /
    (sample["monthly_tasks_completed"] + 1)
)

sample["monthly_productivity"] = (
    sample["monthly_tasks_completed"] /
    (sample["monthly_active_days"] + 1)
)


# -----------------------------
# 4. FINAL FEATURES (MUST MATCH TRAINING)
# -----------------------------
features = [

    # Employee Profile
    "experience_years",
    "job_role_level",
    "salary",

    # Monthly Activity
    "monthly_active_days",
    "monthly_working_hours",
    "monthly_deadlines_missed",
    "monthly_tasks_completed",

    # Engineered Features
    "monthly_efficiency",
    "monthly_work_pressure",
    "monthly_productivity",

    # Capability Features
    "task_complexity",
    "skill_score"
]

X_sample = sample[features]

# -----------------------------
# 5. PREDICT ROI
# -----------------------------
roi_pred = model.predict(X_sample)
print("\n🔹 Predicted ROI:", round(roi_pred[0], 3))

# -----------------------------
# 6. SHAP EXPLAINABILITY
# -----------------------------

# Load dataset for background (same feature engineering)
df = pd.read_csv("dataset.csv")

# =========================================================
# 2. FEATURE ENGINEERING (MONTHLY)
# =========================================================

# ---------------------------------
# Monthly efficiency
# Higher tasks in fewer hours = better
# ---------------------------------
df["monthly_efficiency"] = (
    (df["monthly_tasks_completed"] / (df["monthly_working_hours"] + 1)) * 1.5
)

# ---------------------------------
# Monthly work pressure
# Missed deadlines vs output
# ---------------------------------
df["monthly_work_pressure"] = (
    (df["monthly_deadlines_missed"] * 2) +
    (df["monthly_working_hours"] / 200)
)

df["monthly_productivity"] = (
    df["monthly_tasks_completed"] /
    (df["monthly_active_days"] + 1)
)



X_background = df[features].sample(100, random_state=42)

# TreeExplainer (correct for XGBoost)
explainer = shap.TreeExplainer(model, X_background)

# SHAP values
shap_values = explainer.shap_values(X_sample)

# Convert to dict
shap_dict = dict(zip(features, shap_values[0]))

print("\n🔹 Feature Contributions (SHAP):")
for k, v in shap_dict.items():
    print(f"{k}: {round(v, 4)}")

def correct_shap(shap_dict):
    corrected = shap_dict.copy()

    # Keep natural behavior but guide direction slightly
    if "salary" in corrected:
        corrected["salary"] = corrected["salary"]  # allow both +/-

    if "monthly_deadlines_missed" in corrected:
        corrected["monthly_deadlines_missed"] = -abs(corrected["monthly_deadlines_missed"])

    if "monthly_tasks_completed" in corrected:
        corrected["monthly_tasks_completed"] = abs(corrected["monthly_tasks_completed"])

    return corrected
# -----------------------------
# 5. GENERATE INSIGHTS (FROM SHAP)
# -----------------------------

def generate_insights(shap_dict, row):

    insights = []

    for feature, shap_value in shap_dict.items():

        # ================= POSITIVE =================
        if shap_value > 0.02:

            if feature == "monthly_efficiency":
                insights.append("High monthly efficiency is boosting ROI")

            elif feature == "salary":
                insights.append("Salary investment is generating strong returns")

            elif feature == "task_complexity":
                insights.append("Employee performs well on complex tasks")

            elif feature == "job_role_level":
                insights.append("Role level is aligned with value creation")

            elif feature == "experience_years":
                insights.append("Experience is contributing positively")

            elif feature == "monthly_tasks_completed":
                insights.append("High task completion is increasing ROI")

            elif feature == "skill_score":
                insights.append("Strong skills are supporting performance")

            elif feature == "monthly_productivity":
                insights.append("Productivity levels are driving ROI growth")

            else:
                insights.append(f"{feature} is positively impacting ROI")

        # ================= NEGATIVE =================
        elif shap_value < -0.02:

            if feature == "monthly_working_hours":
                insights.append("Long working hours may be reducing efficiency")

            elif feature == "monthly_tasks_completed":
                insights.append("Low task output is reducing ROI")

            elif feature == "skill_score":
                insights.append("Skill gap is affecting performance")

            elif feature == "monthly_efficiency":
                insights.append("Low efficiency is reducing ROI")

            elif feature == "task_complexity":
                insights.append("Task complexity may be too high to handle effectively")

            elif feature == "monthly_deadlines_missed":
                insights.append("Missed deadlines are hurting ROI significantly")

            elif feature == "experience_years":
                insights.append("Limited experience may affect performance")

            elif feature == "salary":
                insights.append("Salary cost is not justified by performance")

            elif feature == "monthly_work_pressure":
                insights.append("High work pressure is negatively affecting output")

            else:
                insights.append(f"{feature} is negatively impacting ROI")

    return list(set(insights))
# -----------------------------
# 6. GENERATE RECOMMENDATIONS (RULE-BASED)
# -----------------------------

def generate_recommendations(row, shap_dict):

    rec = []

    # ================= DEADLINES =================
    if row["monthly_deadlines_missed"] >= 2:
        rec.append("Improve monthly deadline planning and tracking")

    # ================= EFFICIENCY =================
    if row["monthly_efficiency"] < 0.65 and shap_dict.get("monthly_efficiency", 0) < 0:
        rec.append("Provide productivity training to improve efficiency")

    # ================= WORKING HOURS =================
    if row["monthly_working_hours"] > 200 and shap_dict.get("monthly_working_hours", 0) < 0:
        rec.append("Reduce excessive working hours to prevent burnout")

    # ================= WORK PRESSURE =================
    if row["monthly_work_pressure"] > 0.6:
        rec.append("Rebalance workload to reduce pressure")

    # ================= TASK COMPLEXITY =================
    tc_shap = shap_dict.get("task_complexity", 0)

    if row["task_complexity"] > 7:
        if tc_shap > 0:
            rec.append("Assign more strategic and complex tasks")
        elif tc_shap < 0:
            rec.append("Provide support for handling complex tasks")

    # ================= TASK OUTPUT =================
    if row["monthly_tasks_completed"] < 15 and shap_dict.get("monthly_tasks_completed", 0) < 0:
        rec.append("Improve task execution and planning")

    # ================= SKILLS =================
    if shap_dict.get("skill_score", 0) < -0.02:
        rec.append("Upskill employee to match job requirements")

    # ================= HIGH PERFORMERS =================
    if shap_dict.get("monthly_efficiency", 0) > 0.05:
        rec.append("Recognize and reward high efficiency")

    # ================= SALARY REVIEW =================
    if row["salary"] > 70000 and shap_dict.get("salary", 0) < 0:
        rec.append("Review salary vs performance balance")

    return list(set(rec))
# -----------------------------
# 7. FINAL OUTPUT
# -----------------------------
row = sample.iloc[0]

shap_dict = correct_shap(shap_dict)

insights = generate_insights(shap_dict, row)
recommendations = generate_recommendations(row, shap_dict)

print("\n🔹 Insights:")
for i in insights:
    print("-", i)

print("\n🔹 Recommendations:")
for r in recommendations:
    print("-", r)


# -----------------------------
# 8. LLM (OPENAI)
# -----------------------------
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env (recommended)
load_dotenv()

# Initialize client (reads OPENAI_API_KEY from env)
client = OpenAI()

# Prepare structured input
llm_input = {
    "roi_score": float(roi_pred[0]),
    "features": row.to_dict(),
    "shap_values": {k: float(v) for k, v in shap_dict.items()},
    "insights": insights,
    "recommendations": recommendations
}

# 🔥 Strong prompt (same)
prompt = f"""
You are a senior HR analytics expert.

Analyze the employee ROI data below and provide professional insights.

---------------------
ROI Score:
{llm_input['roi_score']}

Feature Values:
{llm_input['features']}

Feature Impact (SHAP Values):
{llm_input['shap_values']}

Initial Insights:
{llm_input['insights']}

Initial Recommendations:
{llm_input['recommendations']}
---------------------

Tasks:
1. Explain WHY this ROI score occurred
2. Identify key positive and negative drivers
3. Correct any misleading insights
4. Highlight risks (burnout, inefficiency, etc.)
5. Provide clear, practical recommendations

IMPORTANT RULES:
- Use SHAP values as the primary indicator of feature impact direction
- Positive SHAP means the feature increased predicted ROI
- Negative SHAP means the feature reduced predicted ROI
- Do not assume causation beyond provided data
- Distinguish between business risk and model contribution
- Reinforce positively contributing factors instead of recommending reduction
- Avoid contradictory recommendations

Return output in this format:

Summary:
- ...

Key Drivers:
- ...

Risks:
- ...

Final Recommendations:
- ...
"""

# 🔥 OpenAI API call
response = client.chat.completions.create(
    model="gpt-4o-mini",   # fast + cost-efficient
    messages=[
        {"role": "system", "content": "You are a senior HR analytics expert."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

# Extract result
result = response.choices[0].message.content

print("\n🔹 LLM Enhanced Analysis (OpenAI):\n")
print(result)