from __future__ import annotations

"""
test_team_formation.py

Tests the team formation pipeline using MOCK employee data
so you don't need a live MongoDB connection to verify logic.

To test with real MongoDB:
  Set MONGO_URI, MONGO_DB env vars and call recommend_team() directly.
"""

import json
from typing import Any, Dict, List

from employee_scorer import score_all_employees
from team_builder import build_team_options


# =========================
# Mock Employee Data
# (mirrors what MongoDB would return)
# =========================
MOCK_EMPLOYEES: List[Dict[str, Any]] = [
    {
        "_id": "emp001",
        "name": "Aryan Sharma",
        "role": "Full Stack Developer",
        "skills": ["React", "Node.js", "MongoDB", "JavaScript"],
        "performance_score": 88,
        "current_workload": 30,   # 30% busy → 70% available
    },
    {
        "_id": "emp002",
        "name": "Priya Mehta",
        "role": "Backend Developer",
        "skills": ["Node.js", "Python", "MongoDB", "REST API"],
        "performance_score": 75,
        "current_workload": 50,
    },
    {
        "_id": "emp003",
        "name": "Rohan Verma",
        "role": "Frontend Developer",
        "skills": ["React", "CSS", "JavaScript", "TypeScript"],
        "performance_score": 82,
        "current_workload": 20,
    },
    {
        "_id": "emp004",
        "name": "Sneha Patel",
        "role": "DevOps Engineer",
        "skills": ["Docker", "Kubernetes", "AWS", "CI/CD"],
        "performance_score": 90,
        "current_workload": 60,
    },
    {
        "_id": "emp005",
        "name": "Karan Joshi",
        "role": "Full Stack Developer",
        "skills": ["React", "Node.js", "PostgreSQL", "JavaScript"],
        "performance_score": 70,
        "current_workload": 40,
    },
    {
        "_id": "emp006",
        "name": "Meera Nair",
        "role": "UI/UX Designer",
        "skills": ["Figma", "UI", "UX", "CSS", "React"],
        "performance_score": 85,
        "current_workload": 25,
    },
    {
        "_id": "emp007",
        "name": "Amit Gupta",
        "role": "Data Engineer",
        "skills": ["Python", "MongoDB", "SQL", "ETL"],
        "performance_score": 78,
        "current_workload": 55,
    },
    {
        "_id": "emp008",
        "name": "Divya Rao",
        "role": "QA Engineer",
        "skills": ["Testing", "Selenium", "JavaScript", "Node.js"],
        "performance_score": 72,
        "current_workload": 35,
    },
]


def _print_team(team: Dict[str, Any], label: str):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Composite Score : {team['composite_score']}")
    print(f"  Skill Coverage  : {team['skill_coverage']}")
    print(f"  Covered Skills  : {team['covered_skills']}")
    if team["missing_skills"]:
        print(f"  ⚠️  Missing Skills : {team['missing_skills']}")
    print(f"\n  Members:")
    for m in team["members"]:
        print(f"\n    👤 {m['name']} — {m['role']}")
        print(f"       Skills      : {m['skills']}")
        print(f"       Final Score : {m['final_score']}")
        breakdown = m["score_breakdown"]
        print(f"       Breakdown   : skill={breakdown['skill_match']} | "
              f"perf={breakdown['performance']} | "
              f"avail={breakdown['availability']} | "
              f"domain={breakdown['project_type_match']}")


def run_test(
    project_type: str,
    required_skills: List[str],
    team_size: int = 3,
    top_n: int = 3,
):
    print(f"\n🚀 Team Formation Test")
    print(f"   Project Type     : {project_type}")
    print(f"   Required Skills  : {required_skills}")
    print(f"   Team Size        : {team_size}")

    # Score employees
    scored = score_all_employees(MOCK_EMPLOYEES, required_skills, project_type)

    print(f"\n📊 All Employee Scores (ranked):")
    for emp in scored:
        print(f"   {emp['name']:<20} Score: {emp['final_score']}")

    # Build teams
    options = build_team_options(scored, required_skills, team_size=team_size, top_n_options=top_n)

    if not options:
        print("❌ No team options generated.")
        return

    # Print recommended team
    _print_team(options[0], f"✅ RECOMMENDED TEAM (Rank 1)")

    # Print alternatives
    for alt in options[1:]:
        _print_team(alt, f"🔄 ALTERNATIVE TEAM (Rank {alt['rank']})")


if __name__ == "__main__":
    # =========================
    # Test Case 1: Web Development
    # =========================
    run_test(
        project_type="Web Development",
        required_skills=["React", "Node.js", "MongoDB"],
        team_size=3,
        top_n=3,
    )

    # =========================
    # Test Case 2: DevOps
    # =========================
    run_test(
        project_type="DevOps",
        required_skills=["Docker", "Kubernetes", "AWS"],
        team_size=2,
        top_n=2,
    )
