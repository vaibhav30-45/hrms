from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Set


def _team_skill_coverage(team: List[Dict[str, Any]], required_skills: List[str]) -> float:
    """
    What % of required skills are covered by the entire team combined?
    """
    if not required_skills:
        return 1.0

    team_skills: Set[str] = set()
    for member in team:
        for skill in member.get("skills", []):
            team_skills.add(skill.strip().lower())

    req_norm = [s.strip().lower() for s in required_skills]
    covered  = sum(1 for s in req_norm if s in team_skills)
    return round(covered / len(req_norm), 4)


def _team_avg_score(team: List[Dict[str, Any]]) -> float:
    scores = [m["final_score"] for m in team]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _team_composite_score(
    team: List[Dict[str, Any]],
    required_skills: List[str],
    coverage_weight: float = 0.5,
    avg_score_weight: float = 0.5,
) -> float:
    coverage  = _team_skill_coverage(team, required_skills)
    avg_score = _team_avg_score(team)
    return round(coverage * coverage_weight + avg_score * avg_score_weight, 4)


def _covered_skills(team: List[Dict[str, Any]], required_skills: List[str]) -> List[str]:
    team_skills: Set[str] = set()
    for member in team:
        for skill in member.get("skills", []):
            team_skills.add(skill.strip().lower())
    return [s for s in required_skills if s.strip().lower() in team_skills]


def _missing_skills(team: List[Dict[str, Any]], required_skills: List[str]) -> List[str]:
    covered = {s.strip().lower() for s in _covered_skills(team, required_skills)}
    return [s for s in required_skills if s.strip().lower() not in covered]


def build_team_options(
    scored_employees: List[Dict[str, Any]],
    required_skills: List[str],
    team_size: int = 3,
    top_n_options: int = 3,
    candidate_pool: int = 10,       # only consider top N employees for combinations
) -> List[Dict[str, Any]]:
    """
    Generate top N team combinations from scored employees.

    Strategy:
    - Take top `candidate_pool` employees by individual score
    - Generate all combinations of `team_size`
    - Score each team on: skill coverage + avg member score
    - Return top N teams
    """
    # Limit pool to avoid combinatorial explosion
    pool = scored_employees[:candidate_pool]

    if len(pool) < team_size:
        # Not enough employees — return what we have as single team
        team = pool
        return [_format_team_option(team, required_skills, rank=1)]

    # Generate all combinations
    all_combos = list(combinations(pool, team_size))

    # Score each combination
    scored_teams = []
    for combo in all_combos:
        team   = list(combo)
        score  = _team_composite_score(team, required_skills)
        scored_teams.append((score, team))

    # Sort by composite score descending
    scored_teams.sort(key=lambda x: x[0], reverse=True)

    # Return top N options
    options = []
    for rank, (score, team) in enumerate(scored_teams[:top_n_options], start=1):
        options.append(_format_team_option(team, required_skills, rank=rank))

    return options


def _format_team_option(
    team: List[Dict[str, Any]],
    required_skills: List[str],
    rank: int,
) -> Dict[str, Any]:
    covered = _covered_skills(team, required_skills)
    missing = _missing_skills(team, required_skills)
    coverage_pct = round(len(covered) / max(len(required_skills), 1) * 100, 1)

    return {
        "rank":             rank,
        "team_size":        len(team),
        "composite_score":  _team_composite_score(team, required_skills),
        "skill_coverage":   f"{coverage_pct}%",
        "covered_skills":   covered,
        "missing_skills":   missing,
        "members": [
            {
                "employee_id":     m["employee_id"],
                "name":            m["name"],
                "role":            m["role"],
                "skills":          m["skills"],
                "final_score":     m["final_score"],
                "score_breakdown": m["score_breakdown"],
            }
            for m in team
        ],
    }
