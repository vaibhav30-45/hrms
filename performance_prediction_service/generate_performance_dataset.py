import pandas as pd
import numpy as np
from pathlib import Path

# 📁 Save path
MODULE_DIR = Path(__file__).resolve().parent
DATA_PATH = MODULE_DIR / "performance_data.csv"


def generate_dataset(n_samples=3000, random_state=42):
    np.random.seed(random_state)

    data = []

    for _ in range(n_samples):
        # =========================
        # Features (Realistic ranges)
        # NOTE: task_completion_rate and project_success_rate are DECIMALS (0.0–1.0)
        #       attendance is a PERCENTAGE (60–100)
        #       peer_reviews is a RATING (1–5)
        # =========================
        attendance = np.random.randint(50, 101)          # % → 50 to 100
        task_completion = np.random.uniform(0.4, 1.0)   # decimal → 0.4 to 1.0
        peer_reviews = np.random.uniform(1, 5)           # rating → 1 to 5
        project_success = np.random.uniform(0.4, 1.0)   # decimal → 0.4 to 1.0

        # =========================
        # Weighted scoring logic
        # =========================
        score = (
            0.3 * (attendance / 100)
            + 0.3 * task_completion
            + 0.2 * (peer_reviews / 5)
            + 0.2 * project_success
        )

        # Add noise
        noise = np.random.normal(0, 0.05)
        score = np.clip(score + noise, 0, 1)

        # =========================
        # Performance Label
        # =========================
        if score > 0.75:
            performance = "High"
        elif score > 0.52:
            performance = "Medium"
        else:
            performance = "Low"

        # =========================
        # Promotion (tighter threshold → 0.75)
        # =========================
        promotion_prob = score + np.random.normal(0, 0.1)

        if promotion_prob > 0.75:   # FIXED: was 0.7 (too loose)
            promotion = "Yes"
        elif promotion_prob > 0.50: # FIXED: was 0.6 (too loose)
            promotion = "maybe"
        else:
            promotion = "No"

        # =========================
        # Skill Gap (lowered thresholds → better distribution)
        # =========================
        skill_score = 1 - score + np.random.normal(0, 0.1)

        if skill_score > 0.45:      # FIXED: was 0.6 (too hard to reach)
            skill_gap = "High"
        elif skill_score > 0.30:    # FIXED: was 0.4
            skill_gap = "Medium"
        else:
            skill_gap = "Low"

        data.append([
            attendance,
            task_completion,
            peer_reviews,
            project_success,
            performance,
            promotion,
            skill_gap
        ])

    df = pd.DataFrame(data, columns=[
        "attendance",
        "task_completion_rate",
        "peer_reviews",
        "project_success_rate",
        "performance",
        "promotion",
        "skill_gap"
    ])

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(DATA_PATH, index=False)

    print("✅ Dataset generated at:", DATA_PATH)
    print("\nClass Distribution:")
    print("\nPerformance:\n", df["performance"].value_counts())
    print("\nPromotion:\n", df["promotion"].value_counts())
    print("\nSkill Gap:\n", df["skill_gap"].value_counts())


if __name__ == "__main__":
    generate_dataset()
