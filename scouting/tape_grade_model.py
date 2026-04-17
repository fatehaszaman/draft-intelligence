"""
scouting/tape_grade_model.py
════════════════════════════════════════════════════════════════════════════════
Film/Tape Scouting Grade Module

BUSINESS SUMMARY
────────────────
Combine numbers measure how fast a player runs in shorts. Tape grades measure
whether that player actually plays fast in a helmet. This module encodes the
qualitative scouting dimensions that separate elite prospects from elite
athletes — things like hand-fighting technique, pad level, release quickness,
and football IQ signals that only appear on film.

Each factor is scored 1-10 by position coaches and scout consensus, then
weighted into a position-specific composite. Intangibles (leadership,
coachability, work ethic) are scored from combine-week interview proxies
and coaching network reports.

SCORING MODEL
─────────────
Each factor: 1.0 (poor) to 10.0 (elite)
Position-specific weights applied per factor
overall_tape_grade = weighted average of all applicable factors, scaled 0-100

POSITION FACTOR RELEVANCE
─────────────────────────
• QB/WR:   release quickness, route precision, hand fighting, anticipation
• OL/DL:   pad level, leverage, hand fighting, motor
• DB/LB:   change of direction, anticipation, instincts, motor
• All:     change of direction, motor/effort, football IQ proxy
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ── Factor Definitions ────────────────────────────────────────────────────────

# Weights per factor per position group.
# Values sum to 1.0 per position (approximately — normalized in code).
POSITION_TAPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "QB": {
        "release_quickness":    0.25,
        "anticipation_throws":  0.25,
        "football_iq":          0.20,
        "change_of_direction":  0.10,
        "motor_effort":         0.10,
        "pocket_presence":      0.10,
    },
    "WR": {
        "release_quickness":    0.20,
        "route_precision":      0.25,
        "hand_fighting":        0.15,
        "football_iq":          0.15,
        "change_of_direction":  0.15,
        "motor_effort":         0.10,
    },
    "OT": {
        "pad_level":            0.20,
        "hand_fighting":        0.25,
        "leverage_technique":   0.25,
        "motor_effort":         0.10,
        "football_iq":          0.10,
        "change_of_direction":  0.10,
    },
    "OG": {
        "pad_level":            0.25,
        "hand_fighting":        0.25,
        "leverage_technique":   0.25,
        "motor_effort":         0.15,
        "football_iq":          0.10,
    },
    "C": {
        "football_iq":          0.30,  # Line calls, protection identification
        "pad_level":            0.20,
        "hand_fighting":        0.20,
        "leverage_technique":   0.20,
        "motor_effort":         0.10,
    },
    "EDGE": {
        "hand_fighting":        0.25,
        "pad_level":            0.20,
        "motor_effort":         0.20,
        "change_of_direction":  0.15,
        "football_iq":          0.10,
        "leverage_technique":   0.10,
    },
    "DT": {
        "hand_fighting":        0.25,
        "pad_level":            0.25,
        "leverage_technique":   0.20,
        "motor_effort":         0.20,
        "football_iq":          0.10,
    },
    "LB": {
        "football_iq":          0.25,
        "change_of_direction":  0.20,
        "motor_effort":         0.20,
        "hand_fighting":        0.15,
        "anticipation_throws":  0.10,
        "pad_level":            0.10,
    },
    "CB": {
        "hand_fighting":        0.20,
        "change_of_direction":  0.25,
        "football_iq":          0.20,
        "motor_effort":         0.15,
        "release_quickness":    0.10,
        "anticipation_throws":  0.10,
    },
    "S": {
        "football_iq":          0.30,
        "change_of_direction":  0.20,
        "anticipation_throws":  0.20,
        "motor_effort":         0.15,
        "hand_fighting":        0.15,
    },
    "RB": {
        "change_of_direction":  0.30,
        "motor_effort":         0.20,
        "football_iq":          0.15,
        "hand_fighting":        0.15,
        "pad_level":            0.10,
        "route_precision":      0.10,
    },
    "TE": {
        "hand_fighting":        0.15,
        "route_precision":      0.20,
        "football_iq":          0.20,
        "motor_effort":         0.15,
        "change_of_direction":  0.15,
        "release_quickness":    0.15,
    },
}

# Intangibles factors (sourced from combine interview proxies + coaching reports)
INTANGIBLES_FACTORS = [
    "leadership_grade",      # 1-10: recognized leader by coaches/teammates
    "coachability_tape",     # 1-10: adjusts to coaching in-game on tape
    "work_ethic_proxy",      # 1-10: practice film quality vs game film
    "competitiveness",       # 1-10: plays hard in blowouts, late-game effort
    "communication_grade",   # 1-10: combine interview, media composure
]

# Weights for intangibles composite (must sum to 1.0)
INTANGIBLES_WEIGHTS = {
    "leadership_grade":   0.25,
    "coachability_tape":  0.30,
    "work_ethic_proxy":   0.20,
    "competitiveness":    0.15,
    "communication_grade":0.10,
}

# ── Hardcoded Tape Grade Database ─────────────────────────────────────────────
# All grades on 1-10 scale (10 = generational at that factor)
# Sources: PFF scouting reports, NFL Network All-22 analysis, ESPN scout consensus

TAPE_GRADES_DB: Dict[str, Dict] = {
    # ── 2025 Draft Class ──────────────────────────────────────────────────────
    "Cam Ward": {
        "position": "QB",
        "factors": {
            "release_quickness":   8.5,
            "anticipation_throws": 8.0,
            "football_iq":         8.5,
            "change_of_direction": 7.0,
            "motor_effort":        8.5,
            "pocket_presence":     8.0,
        },
        "intangibles": {
            "leadership_grade":    8.0,
            "coachability_tape":   8.2,
            "work_ethic_proxy":    8.0,
            "competitiveness":     9.0,
            "communication_grade": 7.5,
        },
        "red_flags": [],
        "strengths": ["Elite arm talent under pressure", "Exceptional improvisation"],
    },
    "Travis Hunter": {
        "position": "CB",
        "factors": {
            "hand_fighting":       9.5,
            "change_of_direction": 9.5,
            "football_iq":         9.0,
            "motor_effort":        9.5,
            "release_quickness":   9.0,
            "anticipation_throws": 8.5,
        },
        "intangibles": {
            "leadership_grade":    8.5,
            "coachability_tape":   9.0,
            "work_ethic_proxy":    9.5,
            "competitiveness":    10.0,
            "communication_grade": 8.0,
        },
        "red_flags": [],
        "strengths": ["Two-way dominance at elite level", "Generational competitor"],
    },
    "Abdul Carter": {
        "position": "EDGE",
        "factors": {
            "hand_fighting":       9.0,
            "pad_level":           8.5,
            "motor_effort":        9.5,
            "change_of_direction": 9.0,
            "football_iq":         8.5,
            "leverage_technique":  8.0,
        },
        "intangibles": {
            "leadership_grade":    9.0,
            "coachability_tape":   9.2,
            "work_ethic_proxy":    9.0,
            "competitiveness":     9.5,
            "communication_grade": 8.5,
        },
        "red_flags": [],
        "strengths": ["Non-stop motor", "Elite bend around edge", "Natural pass rush moves"],
    },
    "Will Campbell": {
        "position": "OT",
        "factors": {
            "pad_level":           8.5,
            "hand_fighting":       8.0,
            "leverage_technique":  8.5,
            "motor_effort":        8.0,
            "football_iq":         8.5,
            "change_of_direction": 7.5,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   8.5,
            "work_ethic_proxy":    8.0,
            "competitiveness":     8.5,
            "communication_grade": 7.5,
        },
        "red_flags": ["Shorter arms for OT (35.4\") — technique must compensate"],
        "strengths": ["Exceptional hand placement", "Smart anchor vs power rushers"],
    },
    "Mason Graham": {
        "position": "DT",
        "factors": {
            "hand_fighting":       9.0,
            "pad_level":           9.0,
            "leverage_technique":  9.0,
            "motor_effort":        9.5,
            "football_iq":         8.5,
        },
        "intangibles": {
            "leadership_grade":    9.0,
            "coachability_tape":   9.0,
            "work_ethic_proxy":    9.5,
            "competitiveness":     9.5,
            "communication_grade": 8.5,
        },
        "red_flags": [],
        "strengths": ["Dominates interior blocking in every game", "Best run-stuffing DT in class"],
    },
    "Tetairoa McMillan": {
        "position": "WR",
        "factors": {
            "release_quickness":   8.5,
            "route_precision":     9.0,
            "hand_fighting":       8.5,
            "football_iq":         8.0,
            "change_of_direction": 8.0,
            "motor_effort":        8.0,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   8.5,
            "work_ethic_proxy":    8.5,
            "competitiveness":     8.5,
            "communication_grade": 7.5,
        },
        "red_flags": [],
        "strengths": ["Exceptional catch radius", "Makes contested catches look routine"],
    },
    "Malaki Starks": {
        "position": "S",
        "factors": {
            "football_iq":         9.0,
            "change_of_direction": 8.5,
            "anticipation_throws": 8.5,
            "motor_effort":        8.5,
            "hand_fighting":       7.5,
        },
        "intangibles": {
            "leadership_grade":    8.0,
            "coachability_tape":   8.8,
            "work_ethic_proxy":    8.5,
            "competitiveness":     8.5,
            "communication_grade": 8.0,
        },
        "red_flags": [],
        "strengths": ["Georgia-trained football IQ", "Elite pre-snap reads"],
    },
    "Jalon Walker": {
        "position": "EDGE",
        "factors": {
            "hand_fighting":       8.5,
            "pad_level":           8.0,
            "motor_effort":        9.0,
            "change_of_direction": 9.0,
            "football_iq":         8.5,
            "leverage_technique":  7.5,
        },
        "intangibles": {
            "leadership_grade":    8.0,
            "coachability_tape":   8.7,
            "work_ethic_proxy":    9.0,
            "competitiveness":     9.0,
            "communication_grade": 7.5,
        },
        "red_flags": [],
        "strengths": ["Exceptional athleticism off the edge", "Versatile linebacker/edge hybrid"],
    },
    "Darius Alexander": {
        "position": "DT",
        "factors": {
            "hand_fighting":       8.5,
            "pad_level":           8.5,
            "leverage_technique":  8.0,
            "motor_effort":        9.0,
            "football_iq":         7.5,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   8.0,
            "work_ethic_proxy":    8.5,
            "competitiveness":     9.0,
            "communication_grade": 7.0,
        },
        "red_flags": ["MAC conference level — limited elite competition tested"],
        "strengths": ["Dominant vs MAC competition", "High motor every rep"],
    },
    "Mykel Williams": {
        "position": "EDGE",
        "factors": {
            "hand_fighting":       8.0,
            "pad_level":           8.0,
            "motor_effort":        8.5,
            "change_of_direction": 8.5,
            "football_iq":         8.0,
            "leverage_technique":  7.5,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   8.5,
            "work_ethic_proxy":    8.5,
            "competitiveness":     8.5,
            "communication_grade": 7.5,
        },
        "red_flags": [],
        "strengths": ["Long arms", "Georgia pass rush development program"],
    },
    "Jihaad Campbell": {
        "position": "LB",
        "factors": {
            "football_iq":         9.0,
            "change_of_direction": 9.0,
            "motor_effort":        9.5,
            "hand_fighting":       8.0,
            "anticipation_throws": 8.5,
            "pad_level":           7.5,
        },
        "intangibles": {
            "leadership_grade":    8.5,
            "coachability_tape":   8.8,
            "work_ethic_proxy":    9.5,
            "competitiveness":     9.5,
            "communication_grade": 8.0,
        },
        "red_flags": [],
        "strengths": ["ACL recovery work ethic legendary at Alabama", "Elite instincts"],
    },
    "Kelvin Banks Jr": {
        "position": "OT",
        "factors": {
            "pad_level":           8.0,
            "hand_fighting":       8.5,
            "leverage_technique":  8.0,
            "motor_effort":        8.0,
            "football_iq":         8.5,
            "change_of_direction": 7.5,
        },
        "intangibles": {
            "leadership_grade":    8.5,
            "coachability_tape":   8.5,
            "work_ethic_proxy":    8.0,
            "competitiveness":     8.0,
            "communication_grade": 8.0,
        },
        "red_flags": [],
        "strengths": ["Excellent base", "Very few missed assignments on tape"],
    },
    "Tyler Warren": {
        "position": "TE",
        "factors": {
            "hand_fighting":       8.0,
            "route_precision":     9.0,
            "football_iq":         9.5,
            "motor_effort":        9.5,
            "change_of_direction": 8.0,
            "release_quickness":   8.5,
        },
        "intangibles": {
            "leadership_grade":    9.5,
            "coachability_tape":   9.2,
            "work_ethic_proxy":    9.5,
            "competitiveness":     9.0,
            "communication_grade": 9.0,
        },
        "red_flags": [],
        "strengths": ["Walk-on work ethic is genuine and documented", "Routes against man coverage are polished"],
    },
    "Shedeur Sanders": {
        "position": "QB",
        "factors": {
            "release_quickness":   8.0,
            "anticipation_throws": 8.5,
            "football_iq":         8.5,
            "change_of_direction": 6.5,
            "motor_effort":        7.5,
            "pocket_presence":     8.0,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   6.5,
            "work_ethic_proxy":    7.5,
            "competitiveness":     8.0,
            "communication_grade": 6.5,
        },
        "red_flags": [
            "Combine interview flagged as guarded by two teams",
            "Mobility limitations — bottom of QB class athletically",
            "Coachability questions remain open",
        ],
        "strengths": ["Accurate in tight windows", "Poised under pressure from scrutiny"],
    },
    "Omarion Hampton": {
        "position": "RB",
        "factors": {
            "change_of_direction": 8.5,
            "motor_effort":        9.0,
            "football_iq":         8.0,
            "hand_fighting":       7.5,
            "pad_level":           8.5,
            "route_precision":     7.5,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   8.5,
            "work_ethic_proxy":    8.5,
            "competitiveness":     9.0,
            "communication_grade": 7.5,
        },
        "red_flags": [],
        "strengths": ["Contact balance elite", "North-south runner with vision"],
    },
    "Caleb Williams": {
        "position": "QB",
        "factors": {
            "release_quickness":   9.0,
            "anticipation_throws": 8.5,
            "football_iq":         8.5,
            "change_of_direction": 8.0,
            "motor_effort":        7.5,
            "pocket_presence":     8.5,
        },
        "intangibles": {
            "leadership_grade":    7.5,
            "coachability_tape":   7.0,
            "work_ethic_proxy":    7.5,
            "competitiveness":     8.0,
            "communication_grade": 6.5,
        },
        "red_flags": [
            "Independence in scheme preference flagged by scouts",
            "Some reports of disengagement when game plan doesn't feature him prominently",
        ],
        "strengths": ["Ridiculous arm talent", "Extension plays are elite-level"],
    },
    # ── Historical Reference Players ──────────────────────────────────────────
    "Patrick Mahomes": {
        "position": "QB",
        "factors": {
            "release_quickness":  10.0,
            "anticipation_throws":10.0,
            "football_iq":         9.5,
            "change_of_direction": 8.5,
            "motor_effort":        9.0,
            "pocket_presence":    10.0,
        },
        "intangibles": {
            "leadership_grade":    9.5,
            "coachability_tape":   9.8,
            "work_ethic_proxy":   10.0,
            "competitiveness":    10.0,
            "communication_grade": 9.0,
        },
        "red_flags": [],
        "strengths": ["Generational improvisational talent", "Preparation is legendary at every level"],
    },
    "JaMarcus Russell": {
        "position": "QB",
        "factors": {
            "release_quickness":   8.5,
            "anticipation_throws": 5.0,
            "football_iq":         5.5,
            "change_of_direction": 5.0,
            "motor_effort":        4.0,
            "pocket_presence":     6.0,
        },
        "intangibles": {
            "leadership_grade":    4.0,
            "coachability_tape":   3.0,
            "work_ethic_proxy":    3.5,
            "competitiveness":     5.0,
            "communication_grade": 6.0,
        },
        "red_flags": [
            "Practice film inconsistency vs game film — effort gap",
            "Multiple coaches cited failure to study opponent tendencies",
            "Film junkie score: 1/10",
        ],
        "strengths": ["Rare physical arm talent", "Measurables were elite"],
    },
    "Johnny Manziel": {
        "position": "QB",
        "factors": {
            "release_quickness":   7.5,
            "anticipation_throws": 7.0,
            "football_iq":         7.0,
            "change_of_direction": 8.5,
            "motor_effort":        5.0,
            "pocket_presence":     6.0,
        },
        "intangibles": {
            "leadership_grade":    4.0,
            "coachability_tape":   2.0,
            "work_ethic_proxy":    3.0,
            "competitiveness":     6.0,
            "communication_grade": 5.0,
        },
        "red_flags": [
            "Practice attendance issues documented at Cleveland Browns",
            "Off-field distractions visible in preparation quality",
            "Coachability: 2/10 — refuses to adjust footwork or read progressions",
        ],
        "strengths": ["Natural playmaker", "Clutch in Texas A&M — college tape was elite"],
    },
}


class TapeGradeAnalyzer:
    """
    Analyzes film/tape grades for NFL prospects, producing a position-
    adjusted composite grade and intangibles assessment.
    """

    def __init__(self) -> None:
        self._db = TAPE_GRADES_DB
        self._pos_weights = POSITION_TAPE_WEIGHTS
        self._int_weights = INTANGIBLES_WEIGHTS

    # ── Public API ────────────────────────────────────────────────────────────

    def overall_tape_grade(self, player: str) -> float:
        """
        Compute a 0-100 overall tape grade using position-specific factor weights.

        Algorithm:
            1. Retrieve factor scores (1-10) for the player's position
            2. Look up position weight vector; normalize weights to sum 1.0
            3. Weighted average of factor scores → [1, 10]
            4. Scale to [0, 100]: (avg - 1) / 9 × 100

        Returns: float in [0, 100]
        """
        if player not in self._db:
            return 70.0  # Default: neutral grade for unknown players

        data = self._db[player]
        pos = data["position"]
        factors = data["factors"]
        weights = self._pos_weights.get(pos, {})

        if not weights:
            return 70.0

        # Normalize weights to sum 1.0 (handles any rounding drift)
        total_w = sum(weights.values())

        weighted_sum = 0.0
        used_weight = 0.0
        for factor, w in weights.items():
            score = factors.get(factor, 6.5)  # fallback: slightly above average
            weighted_sum += (w / total_w) * score
            used_weight += w / total_w

        avg = weighted_sum / used_weight if used_weight > 0 else 6.5
        # Scale 1-10 → 0-100
        return round(((avg - 1.0) / 9.0) * 100.0, 1)

    def intangibles_score(self, player: str) -> float:
        """
        Compute a 0-100 intangibles composite from combine interview proxies
        and coaching network reports.

        Factors: leadership, coachability (from tape), work ethic proxy,
        competitiveness, communication grade — weighted per INTANGIBLES_WEIGHTS.

        Returns: float in [0, 100]
        """
        if player not in self._db:
            return 70.0

        data = self._db[player]
        intangibles = data.get("intangibles", {})

        if not intangibles:
            return 70.0

        total_w = sum(self._int_weights.values())
        weighted_sum = 0.0
        for factor, w in self._int_weights.items():
            score = intangibles.get(factor, 6.5)
            weighted_sum += (w / total_w) * score

        avg = weighted_sum
        return round(((avg - 1.0) / 9.0) * 100.0, 1)

    def red_flag_check(self, player: str) -> List[str]:
        """
        Return a list of documented concerns from tape review.
        Returns an empty list if no red flags are on record.
        """
        if player not in self._db:
            return []
        return list(self._db[player].get("red_flags", []))

    def strengths(self, player: str) -> List[str]:
        """Return documented tape strengths for a player."""
        if player not in self._db:
            return []
        return list(self._db[player].get("strengths", []))

    def factor_breakdown(self, player: str) -> Dict[str, float]:
        """
        Return the raw factor scores (1-10) for a player, along with
        their normalized weight and contribution to the overall grade.
        """
        if player not in self._db:
            return {}

        data = self._db[player]
        pos = data["position"]
        factors = data["factors"]
        weights = self._pos_weights.get(pos, {})
        total_w = sum(weights.values()) or 1.0

        breakdown = {}
        for factor, w in weights.items():
            score = factors.get(factor, 6.5)
            breakdown[factor] = {
                "score":      score,
                "weight":     round(w / total_w, 3),
                "contribution": round((w / total_w) * score, 3),
            }
        return breakdown

    def leaderboard(self, position: Optional[str] = None) -> List[Dict]:
        """
        Return all players sorted by overall tape grade.
        Optionally filter by position.
        """
        results = []
        for player, data in self._db.items():
            if position and data["position"].upper() != position.upper():
                continue
            grade = self.overall_tape_grade(player)
            intangibles = self.intangibles_score(player)
            flags = self.red_flag_check(player)
            results.append({
                "player":     player,
                "position":   data["position"],
                "tape_grade": grade,
                "intangibles":intangibles,
                "red_flags":  len(flags),
            })
        return sorted(results, key=lambda x: x["tape_grade"], reverse=True)

    def all_players(self) -> List[str]:
        return list(self._db.keys())
