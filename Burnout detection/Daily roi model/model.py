import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("dataset.csv")

# -----------------------------
# 2. SAFE FEATURE ENGINEERING
# -----------------------------


# Efficiency (slightly smoother)
df["efficiency"] = df["tasks_completed"] / (df["working_hours"] + 1)

# Work pressure
df["work_pressure"] = df["deadlines_missed"] / (df["tasks_completed"] + 1)


# -----------------------------
# 3. FEATURES (UPDATED FINAL)
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

X = df[features]
y = df["roi_score"]
# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. MODEL (XGBoost)
# -----------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    # reg_alpha=0.3,   # L1
    # reg_lambda=1.0,  # L2
    subsample=0.5,
    colsample_bytree=0.4,
    random_state=42,
    monotone_constraints=(
    0,   # experience_years
    -1,  # salary ❗
    1,   # job_role_level
    -1,  # working_hours ❗
    -1,  # deadlines_missed
    1,   # tasks_completed ❗
    1,   # efficiency
    -1,  # work_pressure
    1,   # task_complexity
    1    # skill_score
)
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

import pandas as pd

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importance)

import joblib
joblib.dump(model, "model15.pkl")

# import matplotlib.pyplot as plt

# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual ROI")
# plt.ylabel("Predicted ROI")
# plt.title("Actual vs Predicted ROI")
# plt.show()


sample = pd.DataFrame([

    # 1. Overworked Performer
    {
        "experience_years": 3,
        "job_role_level": 2,
        "salary": 40000,
        "working_hours": 14,
        "deadlines_missed": 0,
        "tasks_completed": 10,
        "skill_score": 7,
        "task_complexity": 7
    },

    # 2. Efficient Smart Worker
    {
        "experience_years": 4,
        "job_role_level": 2,
        "salary": 50000,
        "working_hours": 7,
        "deadlines_missed": 0,
        "tasks_completed": 9,
        "skill_score": 8,
        "task_complexity": 7
    },

    # 3. High Salary, Low Output
    {
        "experience_years": 6,
        "job_role_level": 3,
        "salary": 120000,
        "working_hours": 8,
        "deadlines_missed": 2,
        "tasks_completed": 5,
        "skill_score": 6,
        "task_complexity": 5
    },

    # 4. Reliable Low-Cost Worker
    {
        "experience_years": 2,
        "job_role_level": 1,
        "salary": 25000,
        "working_hours": 8,
        "deadlines_missed": 0,
        "tasks_completed": 7,
        "skill_score": 6,
        "task_complexity": 6
    },

    # 5. Burnout Case
    {
        "experience_years": 5,
        "job_role_level": 3,
        "salary": 70000,
        "working_hours": 16,
        "deadlines_missed": 3,
        "tasks_completed": 6,
        "skill_score": 7,
        "task_complexity": 7
    },

    # 6. Top Performer
    {
        "experience_years": 8,
        "job_role_level": 4,
        "salary": 90000,
        "working_hours": 8,
        "deadlines_missed": 0,
        "tasks_completed": 10,
        "skill_score": 9,
        "task_complexity": 9
    },

    # 7. Low Performer
    {
        "experience_years": 4,
        "job_role_level": 2,
        "salary": 60000,
        "working_hours": 6,
        "deadlines_missed": 4,
        "tasks_completed": 3,
        "skill_score": 5,
        "task_complexity": 4
    }

])

# -----------------------------
# FEATURE ENGINEERING (MATCH TRAINING)
# -----------------------------

# Efficiency
sample["efficiency"] = sample["tasks_completed"] / (sample["working_hours"] + 1)

# Work pressure
sample["work_pressure"] = sample["deadlines_missed"] / (sample["tasks_completed"] + 1)


# -----------------------------
# FINAL FEATURES (MUST MATCH TRAINING)
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


# -----------------------------
# PREDICTION
# -----------------------------
predictions = model.predict(sample[features])

print("\n🔹 Predicted ROI:")
for i, roi in enumerate(predictions):
    print(f"Case {i+1}: {round(roi, 3)}")