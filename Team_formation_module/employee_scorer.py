from __future__ import annotations

from typing import Any, Dict, List


# =========================
# Skill match weights
# =========================
WEIGHTS = {
    "skill_match":        0.40,   # how many required skills employee has
    "performance_score":  0.25,   # past performance (0.0–1.0)
    "availability":       0.20,   # current workload (0.0–1.0, higher = more free)
    "project_type_match": 0.15,   # domain/project type familiarity
}

# Project type → related skill keywords
PROJECT_TYPE_SKILL_MAP: Dict[str, List[str]] = {
    "web development":     ["react", "node", "html", "css", "javascript", "typescript", "mongodb", "express"],
    "mobile development":  ["flutter", "react native", "swift", "kotlin", "android", "ios"],
    "data science":        ["python", "ml", "machine learning", "pandas", "numpy", "tensorflow", "pytorch"],
    "devops":              ["docker", "kubernetes", "aws", "ci/cd", "linux", "terraform", "jenkins"],
    "design":              ["figma", "ui", "ux", "adobe", "sketch", "prototyping"],
    "backend":             ["node", "python", "java", "django", "fastapi", "postgresql", "mongodb", "rest api"],
    "frontend":            ["react", "vue", "angular", "html", "css", "javascript", "typescript"],
    "ai/ml":               ["python", "ml", "deep learning", "nlp", "tensorflow", "pytorch", "langchain"],
}


def _normalize_skill(skill: str) -> str:
    return skill.strip().lower()


def _skill_match_score(employee_skills: List[str], required_skills: List[str]) -> float:
    """
    What % of required skills does the employee cover?
    """
    if not required_skills:
        return 1.0

    emp_skills_norm  = {_normalize_skill(s) for s in employee_skills}
    req_skills_norm  = [_normalize_skill(s) for s in required_skills]

    matched = sum(1 for s in req_skills_norm if s in emp_skills_norm)
    return matched / len(req_skills_norm)


def _project_type_match_score(employee_skills: List[str], project_type: str) -> float:
    """
    How well does employee's skillset align with the project domain?
    """
    project_key = project_type.strip().lower()
    domain_skills = PROJECT_TYPE_SKILL_MAP.get(project_key, [])

    if not domain_skills:
        return 0.5   # neutral if unknown project type

    emp_skills_norm   = {_normalize_skill(s) for s in employee_skills}
    domain_skills_norm = [_normalize_skill(s) for s in domain_skills]

    matched = sum(1 for s in domain_skills_norm if s in emp_skills_norm)
    return min(matched / max(len(domain_skills_norm), 1), 1.0)


def _availability_score(current_workload: int) -> float:
    """
    Convert workload % (0–100) to availability score (1.0 = fully free).
    """
    workload = max(0, min(current_workload, 100))
    return round(1.0 - (workload / 100), 2)


def _performance_score(employee: Dict[str, Any]) -> float:
    """
    Normalize performance from employee record.
    Accepts a raw score (0–100) or a category string.
    """
    raw = employee.get("performance_score") or employee.get("performance")

    if isinstance(raw, (int, float)):
        return round(min(max(float(raw), 0), 100) / 100, 2)

    # Handle string categories
    mapping = {"high": 1.0, "medium": 0.6, "low": 0.3}
    return mapping.get(str(raw).strip().lower(), 0.5)


def score_employee(
    employee: Dict[str, Any],
    required_skills: List[str],
    project_type: str,
) -> Dict[str, Any]:
    """
    Score a single employee against project requirements.
    Returns employee data + individual scores + final weighted score.
    """
    emp_skills = employee.get("skills", [])

    skill_match        = _skill_match_score(emp_skills, required_skills)
    performance        = _performance_score(employee)
    availability       = _availability_score(employee.get("current_workload", 50))
    project_type_match = _project_type_match_score(emp_skills, project_type)

    final_score = round(
        WEIGHTS["skill_match"]        * skill_match
        + WEIGHTS["performance_score"]  * performance
        + WEIGHTS["availability"]       * availability
        + WEIGHTS["project_type_match"] * project_type_match,
        4,
    )

    return {
        "employee_id":   str(employee.get("_id", employee.get("id", "unknown"))),
        "name":          employee.get("name", "Unknown"),
        "role":          employee.get("role", "Unknown"),
        "skills":        emp_skills,
        "score_breakdown": {
            "skill_match":        round(skill_match, 4),
            "performance":        round(performance, 4),
            "availability":       round(availability, 4),
            "project_type_match": round(project_type_match, 4),
        },
        "final_score": final_score,
    }


def score_all_employees(
    employees: List[Dict[str, Any]],
    required_skills: List[str],
    project_type: str,
) -> List[Dict[str, Any]]:
    """
    Score and rank all employees. Returns sorted list (highest score first).
    """
    scored = [
        score_employee(emp, required_skills, project_type)
        for emp in employees
    ]
    return sorted(scored, key=lambda x: x["final_score"], reverse=True)
