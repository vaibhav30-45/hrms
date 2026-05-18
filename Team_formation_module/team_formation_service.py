from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from employee_scorer import score_all_employees
from team_builder import build_team_options

LOGGER = logging.getLogger(__name__)

# =========================
# MongoDB Config
# Pull from env or fallback to default
# =========================
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB         = os.getenv("MONGO_DB", "hrms")
EMPLOYEES_COLLECTION = os.getenv("EMPLOYEES_COLLECTION", "employees")


class TeamFormationError(RuntimeError):
    """Raised when team formation fails."""


# =========================
# MongoDB Connection
# =========================
def _get_collection() -> Collection:
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    return db[EMPLOYEES_COLLECTION]


def _fetch_employees(required_skills: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch employees from MongoDB.

    Strategy:
    - First try to find employees with at least one matching skill
    - If too few results, fetch all active employees as fallback
    """
    collection = _get_collection()

    # Normalize skills for case-insensitive match
    skills_normalized = [s.strip().lower() for s in required_skills]

    # Query: active employees who have at least 1 required skill
    query = {
        "status": {"$in": ["active", "Active"]},   # only active employees
        "skills": {
            "$elemMatch": {
                "$regex": "|".join(skills_normalized),
                "$options": "i",
            }
        }
    }

    employees = list(collection.find(query, {"_id": 1, "name": 1, "role": 1,
                                              "skills": 1, "performance_score": 1,
                                              "performance": 1, "current_workload": 1}))

    # Fallback: if less than 5 matched, fetch all active employees
    if len(employees) < 5:
        LOGGER.warning("Few skill-matched employees found (%d). Fetching all active employees.", len(employees))
        fallback_query = {"status": {"$in": ["active", "Active"]}}
        employees = list(collection.find(fallback_query, {
            "_id": 1, "name": 1, "role": 1,
            "skills": 1, "performance_score": 1,
            "performance": 1, "current_workload": 1
        }))

    # Convert ObjectId to string
    for emp in employees:
        if "_id" in emp:
            emp["_id"] = str(emp["_id"])

    LOGGER.info("Fetched %d employees from MongoDB.", len(employees))
    return employees


# =========================
# Core Service Function
# =========================
def recommend_team(
    project_type: str,
    required_skills: List[str],
    team_size: int = 3,
    top_n_options: int = 3,
) -> Dict[str, Any]:
    """
    Main entry point.

    Args:
        project_type:    e.g. "Web Development", "Data Science"
        required_skills: e.g. ["React", "Node.js", "MongoDB"]
        team_size:       how many members per team (default 3)
        top_n_options:   how many alternative teams to return (default 3)

    Returns:
        {
            "status": "success",
            "project": { ... },
            "recommended_team": { rank 1 team },
            "alternatives": [ rank 2, rank 3, ... ],
            "all_employee_scores": [ scored + ranked list of all employees ]
        }
    """
    try:
        if not required_skills:
            raise ValueError("required_skills cannot be empty.")
        if not project_type:
            raise ValueError("project_type cannot be empty.")
        if team_size < 1:
            raise ValueError("team_size must be at least 1.")

        # Step 1: Fetch employees from MongoDB
        employees = _fetch_employees(required_skills)

        if not employees:
            raise TeamFormationError("No active employees found in the database.")

        # Step 2: Score every employee
        scored_employees = score_all_employees(employees, required_skills, project_type)

        # Step 3: Build team combinations
        team_options = build_team_options(
            scored_employees,
            required_skills,
            team_size=team_size,
            top_n_options=top_n_options,
        )

        if not team_options:
            raise TeamFormationError("Could not generate any team options.")

        # Step 4: Separate best team from alternatives
        recommended = team_options[0]
        alternatives = team_options[1:]

        return {
            "status": "success",
            "project": {
                "project_type":    project_type,
                "required_skills": required_skills,
                "team_size":       team_size,
            },
            "recommended_team":    recommended,
            "alternatives":        alternatives,
            "all_employee_scores": [
                {
                    "employee_id":     e["employee_id"],
                    "name":            e["name"],
                    "role":            e["role"],
                    "final_score":     e["final_score"],
                    "score_breakdown": e["score_breakdown"],
                }
                for e in scored_employees
            ],
        }

    except TeamFormationError:
        raise
    except Exception as error:
        LOGGER.error("Team formation failed: %s", error)
        raise TeamFormationError(f"Team formation failed: {error}") from error
