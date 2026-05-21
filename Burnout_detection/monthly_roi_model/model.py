import pandas as pd
import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
import joblib

# =========================================================
# 1. LOAD MONTHLY DATASET
# =========================================================

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
# =========================================================
# 3. FEATURES
# =========================================================

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

X = df[features]
y = df["roi_score"]

# =========================================================
# 4. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 5. XGBOOST MODEL (OPTIMIZED FOR MONTHLY ROI)
# =========================================================

model = XGBRegressor(

    # Core
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,

    # Regularization
    reg_alpha=0.5,
    reg_lambda=1.5,

    # Sampling
    subsample=0.8,
    colsample_bytree=0.8,

    # # Stability
    # min_child_weight=3,
    # gamma=0.1,

    # Random
    random_state=42,

    # Better generalization
    objective="reg:squarederror",

    # Monotonic behavior
    monotone_constraints=(

        # Positive impact
        3,   # experience_years
        3,   # job_role_level

        # Negative cost impact
        2,  # salary

        # Positive
        1,   # monthly_active_days

        # Slight negative (overwork)
        1,  # monthly_working_hours

        # Negative
        -1,  # monthly_deadlines_missed

        # Positive
        1,   # monthly_tasks_completed

        # Positive
        1,   # monthly_efficiency

        # Negative
        -1,  # monthly_work_pressure

       

        # Positive
        1,   # task_complexity

        # Positive
        1    # skill_score
    )
)

# =========================================================
# 6. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# 7. EVALUATION
# =========================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MONTHLY ROI MODEL PERFORMANCE")
print("==============================")

print(f"MAE       : {round(mae, 4)}")
print(f"RMSE      : {round(rmse, 4)}")
print(f"R2 Score  : {round(r2, 4)}")

# =========================================================
# 8. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance)

# =========================================================
# 9. SAVE MODEL
# =========================================================

joblib.dump(model, "monthly_roi_model.pkl")

print("\nModel saved as monthly_roi_model.pkl")

# =========================================================
# 10. SAMPLE PREDICTIONS
# =========================================================

sample = pd.DataFrame([

    # -----------------------------------
    # 1. High Performer
    # -----------------------------------
    {
    "experience_years": 7,
    "job_role_level": 4,
    "salary": 90000,
    "monthly_active_days": 25,
    "monthly_working_hours": 200,
    "monthly_deadlines_missed": 1,
    "monthly_tasks_completed": 190,
    "skill_score": 9,
    "task_complexity": 8
},

    {
    "experience_years": 5,
    "job_role_level": 3,
    "salary": 75000,
    "monthly_active_days": 28,
    "monthly_working_hours": 310,
    "monthly_deadlines_missed": 7,
    "monthly_tasks_completed": 120,
    "skill_score": 7,
    "task_complexity": 7
},

    {
    "experience_years": 2,
    "job_role_level": 1,
    "salary": 40000,
    "monthly_active_days": 20,
    "monthly_working_hours": 160,
    "monthly_deadlines_missed": 9,
    "monthly_tasks_completed": 60,
    "skill_score": 4,
    "task_complexity": 4
},

    {
    "experience_years": 4,
    "job_role_level": 2,
    "salary": 60000,
    "monthly_active_days": 23,
    "monthly_working_hours": 185,
    "monthly_deadlines_missed": 3,
    "monthly_tasks_completed": 120,
    "skill_score": 6,
    "task_complexity": 6
},
{
    "experience_years": 5,
    "job_role_level": 3,
    "salary": 70000,
    "monthly_active_days": 22,
    "monthly_working_hours": 150,
    "monthly_deadlines_missed": 0,
    "monthly_tasks_completed": 170,
    "skill_score": 8,
    "task_complexity": 7
},
{
    "experience_years": 6,
    "job_role_level": 3,
    "salary": 85000,
    "monthly_active_days": 24,
    "monthly_working_hours": 210,
    "monthly_deadlines_missed": 6,
    "monthly_tasks_completed": 100,
    "skill_score": 9,
    "task_complexity": 8
},
{
    "experience_years": 3,
    "job_role_level": 2,
    "salary": 55000,
    "monthly_active_days": 24,
    "monthly_working_hours": 190,
    "monthly_deadlines_missed": 2,
    "monthly_tasks_completed": 140,
    "skill_score": 7,
    "task_complexity": 6
},
{
    "experience_years": 4,
    "job_role_level": 2,
    "salary": 50000,
    "monthly_active_days": 21,
    "monthly_working_hours": 140,
    "monthly_deadlines_missed": 4,
    "monthly_tasks_completed": 90,
    "skill_score": 5,
    "task_complexity": 4
}
])

# =========================================================
# 11. FEATURE ENGINEERING FOR SAMPLE
# =========================================================

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


# =========================================================
# 12. PREDICTIONS
# =========================================================

predictions = model.predict(sample[features])

print("\n==============================")
print("PREDICTED MONTHLY ROI")
print("==============================")

for i, roi in enumerate(predictions):
    print(f"Case {i+1}: {round(float(roi), 3)}")