from performance_prediction_service import predict_employee_performance


def run_tests():
    print("🚀 Testing Performance Prediction Module\n")

    # ✅ High Performer
    # FIXED: task_completion_rate and project_success_rate are DECIMALS (0.0–1.0)
    high_data = {
        "attendance":           95,    
        "task_completion_rate": 0.92,  
        "peer_reviews":         5,     
        "project_success_rate": 0.90,  
    }

    # ✅ Medium Performer
    medium_data = {
        "attendance":           70,
        "task_completion_rate": 0.65,  
        "peer_reviews":         3,
        "project_success_rate": 0.68,   
    }

    # ✅ Low Performer
    low_data = {
        "attendance":           55,
        "task_completion_rate": 0.45,        
        "peer_reviews":         1.5,
        "project_success_rate": 0.45,   
    }

    print("High Performer:  ", predict_employee_performance(high_data))
    print("Medium Performer:", predict_employee_performance(medium_data))
    print("Low Performer:   ", predict_employee_performance(low_data))

    # =========================
    # Edge Case Tests
    # =========================
    print("\n--- Edge Cases ---\n")

    # Border High/Medium
    border_hm = {
        "attendance":           80,
        "task_completion_rate": 0.76,
        "peer_reviews":         3.5,
        "project_success_rate": 0.76,
    }
    print("Border High/Medium:", predict_employee_performance(border_hm))

    # Border Medium/Low
    border_ml = {
        "attendance":           65,
        "task_completion_rate": 0.55,
        "peer_reviews":         2.5,
        "project_success_rate": 0.55,
    }
    print("Border Medium/Low: ", predict_employee_performance(border_ml))


if __name__ == "__main__":
    run_tests()
