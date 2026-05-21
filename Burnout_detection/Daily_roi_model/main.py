import pandas as pd
import numpy as np
import joblib
import shap
from openai import OpenAI
import json
from dotenv import load_dotenv
# -----------------------------
# 1. LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")   # your saved model

# -----------------------------
# 2. SAMPLE INPUT (RAW)
# -----------------------------
sample = pd.DataFrame([{
    
    "experience_years": 5,
    "job_role_level": 2,
    "salary": 70000,
    "working_hours": 14,
    "deadlines_missed": 5,
    "tasks_completed": 6,
    "skill_score": 7,
    "task_complexity": 8

}])

# -----------------------------
# 3. FEATURE ENGINEERING (MATCH TRAINING)
# -----------------------------

# Efficiency
sample["efficiency"] = sample["tasks_completed"] / (sample["working_hours"] + 1)

# Work pressure
sample["work_pressure"] = sample["deadlines_missed"] / (sample["tasks_completed"] + 1)


# -----------------------------
# 4. FINAL FEATURES (MUST MATCH TRAINING)
# -----------------------------
features = [
    "experience_years",
    "salary",
    "job_role_level",
    "working_hours",
    "deadlines_missed",
    "tasks_completed",
    "efficiency",
    "work_pressure",
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

df["efficiency"] = df["tasks_completed"] / (df["working_hours"] + 1)
df["work_pressure"] = df["deadlines_missed"] / (df["tasks_completed"] + 1)


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

    # Salary always negative
    corrected["salary"] = -abs(corrected["salary"])

    # Deadlines always negative
    corrected["deadlines_missed"] = -abs(corrected["deadlines_missed"])

    # Tasks always positive
    corrected["tasks_completed"] = abs(corrected["tasks_completed"])

    return corrected

shap_dict = correct_shap(shap_dict)
# -----------------------------
# 5. GENERATE INSIGHTS (FROM SHAP)
# -----------------------------

def generate_insights(shap_dict, row):

    insights = []

    for feature, shap_value in shap_dict.items():

        # ==================================================
        # POSITIVE CONTRIBUTION
        # ==================================================
        if shap_value > 0.02:

            if feature == "efficiency":
                insights.append(
                    "Strong efficiency is improving overall productivity and ROI"
                )

            elif feature == "salary":
                insights.append(
                    "Current salary investment is generating positive ROI contribution"
                )

            elif feature == "task_complexity":
                insights.append(
                    "Employee performs well on complex and challenging tasks"
                )

            elif feature == "job_role_level":
                insights.append(
                    "Role responsibilities are positively contributing to business value"
                )

            elif feature == "experience_years":
                insights.append(
                    "Experience level is contributing positively to performance"
                )

            elif feature == "tasks_completed":
                insights.append(
                    "Task completion rate is positively impacting ROI"
                )

            elif feature == "skill_score":
                insights.append(
                    "Skill level is supporting strong work performance"
                )

            else:
                insights.append(
                    f"{feature} is positively contributing to ROI"
                )

        # ==================================================
        # NEGATIVE CONTRIBUTION
        # ==================================================
        elif shap_value < -0.02:

            if feature == "working_hours":
                insights.append(
                    "Extended working hours may be reducing productivity and increasing fatigue"
                )

            elif feature == "tasks_completed":
                insights.append(
                    "Low task output is limiting ROI generation"
                )

            elif feature == "skill_score":
                insights.append(
                    "Current skill level may not fully match task requirements"
                )

            elif feature == "efficiency":
                insights.append(
                    "Lower efficiency levels are reducing overall ROI"
                )

            elif feature == "task_complexity":
                insights.append(
                    "Current task complexity may be overwhelming the employee"
                )

            elif feature == "deadlines_missed":
                insights.append(
                    "Missed deadlines are reducing delivery reliability"
                )

            elif feature == "experience_years":
                insights.append(
                    "Early-career experience profile may require additional support"
                )

            elif feature == "salary":
                insights.append(
                    "Salary cost is currently outweighing performance contribution"
                )

            else:
                insights.append(
                    f"{feature} is negatively contributing to ROI"
                )

    return list(set(insights))

# -----------------------------
# 6. GENERATE RECOMMENDATIONS (RULE-BASED)
# -----------------------------

def generate_recommendations(row, shap_dict):

    rec = []

    # ==================================================
    # DEADLINES
    # ==================================================
    if row["deadlines_missed"] >= 2:
        rec.append(
            "Improve planning and deadline management processes"
        )

    # ==================================================
    # EFFICIENCY
    # ==================================================
    if row["efficiency"] < 0.65 and shap_dict.get("efficiency", 0) < 0:
        rec.append(
            "Provide workflow or productivity training to improve efficiency"
        )

    # ==================================================
    # WORKING HOURS
    # ==================================================
    if row["working_hours"] > 9 and shap_dict.get("working_hours", 0) < 0:
        rec.append(
            "Optimize workload and working hours to reduce burnout risk"
        )

    # ==================================================
    # WORK PRESSURE
    # ==================================================
    if row["work_pressure"] > 0.5:
        rec.append(
            "Rebalance workload to reduce sustained pressure"
        )

    # ==================================================
    # TASK COMPLEXITY
    # ==================================================
    tc_shap = shap_dict.get("task_complexity", 0)

    if row["task_complexity"] > 7:

        if tc_shap > 0:
            rec.append(
                "Employee handles complex tasks effectively — consider strategic responsibilities"
            )

        elif tc_shap < 0:
            rec.append(
                "Provide support for managing highly complex tasks"
            )

    # ==================================================
    # TASK OUTPUT
    # ==================================================
    if row["tasks_completed"] < 5 and shap_dict.get("tasks_completed", 0) < 0:
        rec.append(
            "Improve execution planning to increase task completion"
        )

    # ==================================================
    # SKILLS
    # ==================================================
    if shap_dict.get("skill_score", 0) < -0.02:
        rec.append(
            "Upskill employee to better align with job requirements"
        )

    # ==================================================
    # HIGH PERFORMERS
    # ==================================================
    if shap_dict.get("efficiency", 0) > 0.05:
        rec.append(
            "Recognize and reward strong efficiency performance"
        )

    # ==================================================
    # SALARY REVIEW
    # ==================================================
    if (
        row["salary"] > 70000
        and shap_dict.get("salary", 0) < 0
    ):
        rec.append(
            "Review compensation against performance contribution"
        )

    return list(set(rec))
# -----------------------------
# 7. FINAL OUTPUT
# -----------------------------
row = sample.iloc[0]

insights = generate_insights(shap_dict, row)
recommendations = generate_recommendations(row, shap_dict)

print("\n🔹 Insights:")
for i in insights:
    print("-", i)

print("\n🔹 Recommendations:")
for r in recommendations:
    print("-", r)

# -----------------------------
# 8. LLM ENHANCEMENT (FINAL LAYER)
# -----------------------------

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

# Strong prompt (same as before)
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
    model="gpt-4o-mini", 
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