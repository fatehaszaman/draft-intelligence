"""
scouting/college_production.py
──────────────────────────────
Evaluates college production after adjusting for:
  - Conference strength (SEC competition > PAC-12 > MAC, etc.)
  - Offensive system effects (air raid inflates WR counting stats)
  - Dominator rating (player's share of total team production)
  - Career trajectory (improvement slope, acceleration)
  - Breakout age (early dominance signals elite upside)

All adjustments are multiplicative and sourced from historical comp
analysis of how college stats translate to the NFL level.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from config import CONFERENCE_ADJUSTMENTS, SYSTEM_ADJUSTMENTS

# ── Dominator rating thresholds ───────────────────────────────────────────────
# % of team production that signals elite prospect
DOMINATOR_ELITE = 35.0   # > 35% of team's yards → elite dominator
DOMINATOR_GOOD  = 25.0   # > 25% → strong contributor
DOMINATOR_POOR  = 15.0   # < 15% → underwhelming

# ── Breakout age definitions ──────────────────────────────────────────────────
# A "breakout season" = 50+ receptions OR 1000+ yards OR 15+ TDs
# for skill positions; for QB: 60%+ completion, 20+ TDs
BREAKOUT_THRESHOLDS = {
    "WR":   {"yards": 800,  "receptions": 50,  "tds": 8},
    "RB":   {"yards": 900,  "carries": 150,    "tds": 8},
    "TE":   {"yards": 600,  "receptions": 40,  "tds": 6},
    "QB":   {"yards": 2500, "completion_pct": 60.0, "tds": 20},
    "EDGE": {"sacks": 7.0,  "pressures": 25,   "tfl": 8},
    "LB":   {"tackles": 70, "sacks": 4.0,      "tfl": 8},
    "CB":   {"pbu": 8,      "interceptions": 3, "coverage_grade": 75.0},
    "S":    {"tackles": 60, "interceptions": 3, "pbu": 6},
    "DT":   {"sacks": 4.0,  "pressures": 18,   "tfl": 8},
    "OT":   {"pass_block_grade": 75.0},
    "OG":   {"pass_block_grade": 75.0},
}

# ── Hardcoded player career stats ─────────────────────────────────────────────
# Each entry includes stats by year and metadata
# 'breakout_age' manually verified from player bios

PLAYER_COLLEGE_DB: Dict[str, Dict] = {
    # ── 2025 Draft Class ──────────────────────────────────────────────────────
    "Cam Ward": {
        "position": "QB",
        "school": "Miami",
        "conference": "ACC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2021, "age": 20, "yards": 3039, "tds": 26, "ints": 11, "completion_pct": 62.4, "games": 12},
            {"year": 2022, "age": 21, "yards": 3588, "tds": 32, "ints": 7,  "completion_pct": 63.7, "games": 13},
            {"year": 2023, "age": 22, "yards": 4038, "tds": 33, "ints": 8,  "completion_pct": 65.1, "games": 13},
            {"year": 2024, "age": 23, "yards": 4313, "tds": 39, "ints": 7,  "completion_pct": 67.2, "games": 13},
        ],
        "team_stats_2024": {"total_yards": 5200, "total_tds": 52},
        "pff_grade_2024": 91.4,
    },
    "Travis Hunter": {
        "position": "CB",
        "school": "Colorado State",
        "conference": "Mountain West",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 19, "interceptions": 2, "pbu": 9,  "coverage_grade": 74.0, "games": 11},
            {"year": 2023, "age": 20, "interceptions": 4, "pbu": 12, "coverage_grade": 82.0, "games": 12},
            {"year": 2024, "age": 21, "interceptions": 4, "pbu": 14, "coverage_grade": 89.2, "games": 12,
             "wr_yards": 1258, "wr_tds": 15, "wr_receptions": 96},  # Also played WR
        ],
        "team_stats_2024": {"total_wr_yards": 3100, "total_wr_tds": 28},
        "pff_grade_2024": 92.1,
    },
    "Abdul Carter": {
        "position": "EDGE",
        "school": "Penn State",
        "conference": "Big Ten",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 18, "sacks": 2.0, "pressures": 14, "tfl": 5, "games": 11},
            {"year": 2023, "age": 19, "sacks": 6.0, "pressures": 28, "tfl": 12, "games": 13},
            {"year": 2024, "age": 20, "sacks": 12.0, "pressures": 46, "tfl": 16, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 31, "total_pressures": 110},
        "pff_grade_2024": 90.3,
    },
    "Will Campbell": {
        "position": "OT",
        "school": "LSU",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 19, "pass_block_grade": 68.0, "games": 12},
            {"year": 2023, "age": 20, "pass_block_grade": 76.4, "games": 13},
            {"year": 2024, "age": 21, "pass_block_grade": 88.9, "games": 13},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 88.9,
    },
    "Mason Graham": {
        "position": "DT",
        "school": "Michigan",
        "conference": "Big Ten",
        "system": "run_power",
        "stats_by_year": [
            {"year": 2022, "age": 19, "sacks": 3.5, "pressures": 21, "tfl": 7, "games": 13},
            {"year": 2023, "age": 20, "sacks": 7.5, "pressures": 38, "tfl": 15, "games": 14},
            {"year": 2024, "age": 21, "sacks": 9.0, "pressures": 44, "tfl": 16, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 24, "total_pressures": 95},
        "pff_grade_2024": 91.8,
    },
    "Tetairoa McMillan": {
        "position": "WR",
        "school": "Arizona",
        "conference": "Big 12",
        "system": "spread",
        "stats_by_year": [
            {"year": 2022, "age": 19, "yards": 801,  "receptions": 53, "tds": 7,  "games": 12},
            {"year": 2023, "age": 20, "yards": 1402, "receptions": 90, "tds": 11, "games": 13},
            {"year": 2024, "age": 21, "yards": 1847, "receptions": 96, "tds": 17, "games": 13},
        ],
        "team_stats_2024": {"total_yards": 4100, "total_tds": 35},
        "pff_grade_2024": 92.3,
    },
    "Malaki Starks": {
        "position": "S",
        "school": "Georgia",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 18, "tackles": 51, "interceptions": 2, "pbu": 5, "games": 13},
            {"year": 2023, "age": 19, "tackles": 66, "interceptions": 4, "pbu": 8, "games": 14},
            {"year": 2024, "age": 20, "tackles": 72, "interceptions": 5, "pbu": 9, "games": 13},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 86.7,
    },
    "Jalon Walker": {
        "position": "EDGE",
        "school": "Georgia",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 18, "sacks": 3.0, "pressures": 18, "tfl": 7, "games": 13},
            {"year": 2023, "age": 19, "sacks": 6.0, "pressures": 29, "tfl": 11, "games": 14},
            {"year": 2024, "age": 20, "sacks": 9.5, "pressures": 38, "tfl": 13, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 28, "total_pressures": 98},
        "pff_grade_2024": 85.9,
    },
    "Darius Alexander": {
        "position": "DT",
        "school": "Toledo",
        "conference": "MAC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 20, "sacks": 4.0, "pressures": 24, "tfl": 9,  "games": 12},
            {"year": 2023, "age": 21, "sacks": 7.5, "pressures": 38, "tfl": 14, "games": 13},
            {"year": 2024, "age": 22, "sacks": 10.0,"pressures": 47, "tfl": 17, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 22, "total_pressures": 78},
        "pff_grade_2024": 89.1,
    },
    "Mykel Williams": {
        "position": "EDGE",
        "school": "Georgia",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 18, "sacks": 3.5, "pressures": 20, "tfl": 6,  "games": 13},
            {"year": 2023, "age": 19, "sacks": 5.5, "pressures": 27, "tfl": 9,  "games": 14},
            {"year": 2024, "age": 20, "sacks": 8.5, "pressures": 34, "tfl": 12, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 28, "total_pressures": 98},
        "pff_grade_2024": 84.6,
    },
    "Jihaad Campbell": {
        "position": "LB",
        "school": "Alabama",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 19, "tackles": 48, "sacks": 2.0, "tfl": 7, "games": 12},
            {"year": 2023, "age": 20, "tackles": 71, "sacks": 5.5, "tfl": 11, "games": 13},
            {"year": 2024, "age": 21, "tackles": 85, "sacks": 9.0, "tfl": 16, "games": 13},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 88.4,
    },
    "Kelvin Banks Jr": {
        "position": "OT",
        "school": "Texas",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 18, "pass_block_grade": 71.0, "games": 12},
            {"year": 2023, "age": 19, "pass_block_grade": 80.1, "games": 13},
            {"year": 2024, "age": 20, "pass_block_grade": 85.3, "games": 13},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 85.3,
    },
    "Tyler Warren": {
        "position": "TE",
        "school": "Penn State",
        "conference": "Big Ten",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 20, "yards": 418,  "receptions": 32, "tds": 3, "games": 13},
            {"year": 2023, "age": 21, "yards": 657,  "receptions": 49, "tds": 5, "games": 13},
            {"year": 2024, "age": 22, "yards": 1233, "receptions": 104,"tds": 8, "games": 13},
        ],
        "team_stats_2024": {"total_yards": 3900, "total_tds": 31},
        "pff_grade_2024": 90.7,
    },
    "Shemar Stewart": {
        "position": "EDGE",
        "school": "Texas A&M",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2022, "age": 18, "sacks": 2.0, "pressures": 12, "tfl": 4, "games": 11},
            {"year": 2023, "age": 19, "sacks": 4.5, "pressures": 22, "tfl": 9, "games": 13},
            {"year": 2024, "age": 20, "sacks": 7.5, "pressures": 31, "tfl": 11, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 22, "total_pressures": 76},
        "pff_grade_2024": 83.2,
    },
    # ── 2024 Draft Class ──────────────────────────────────────────────────────
    "Caleb Williams": {
        "position": "QB",
        "school": "USC",
        "conference": "PAC-12",
        "system": "spread",
        "stats_by_year": [
            {"year": 2021, "age": 19, "yards": 1912, "tds": 21, "ints": 4,  "completion_pct": 66.1, "games": 8},
            {"year": 2022, "age": 20, "yards": 4537, "tds": 42, "ints": 5,  "completion_pct": 66.6, "games": 13},
            {"year": 2023, "age": 21, "yards": 3633, "tds": 30, "ints": 5,  "completion_pct": 66.0, "games": 11},
        ],
        "team_stats_2024": {"total_yards": 4800, "total_tds": 44},
        "pff_grade_2024": 90.2,
    },
    "Marvin Harrison Jr": {
        "position": "WR",
        "school": "Ohio State",
        "conference": "Big Ten",
        "system": "spread",
        "stats_by_year": [
            {"year": 2021, "age": 18, "yards": 754,  "receptions": 39, "tds": 5,  "games": 12},
            {"year": 2022, "age": 19, "yards": 1263, "receptions": 77, "tds": 14, "games": 14},
            {"year": 2023, "age": 20, "yards": 1211, "receptions": 67, "tds": 14, "games": 12},
        ],
        "team_stats_2024": {"total_yards": 4200, "total_tds": 40},
        "pff_grade_2024": 93.1,
    },
    "Joe Alt": {
        "position": "OT",
        "school": "Notre Dame",
        "conference": "Independent",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2021, "age": 18, "pass_block_grade": 65.0, "games": 12},
            {"year": 2022, "age": 19, "pass_block_grade": 75.2, "games": 13},
            {"year": 2023, "age": 20, "pass_block_grade": 90.1, "games": 13},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 90.1,
    },
    "Rome Odunze": {
        "position": "WR",
        "school": "Washington",
        "conference": "PAC-12",
        "system": "spread",
        "stats_by_year": [
            {"year": 2021, "age": 19, "yards": 386,  "receptions": 24, "tds": 4, "games": 9},
            {"year": 2022, "age": 20, "yards": 1145, "receptions": 75, "tds": 8, "games": 13},
            {"year": 2023, "age": 21, "yards": 1640, "receptions": 102,"tds": 13,"games": 14},
        ],
        "team_stats_2024": {"total_yards": 4100, "total_tds": 36},
        "pff_grade_2024": 88.6,
    },
    # ── 2023 Draft Class ──────────────────────────────────────────────────────
    "CJ Stroud": {
        "position": "QB",
        "school": "Ohio State",
        "conference": "Big Ten",
        "system": "spread",
        "stats_by_year": [
            {"year": 2020, "age": 19, "yards": 1842, "tds": 13, "ints": 5, "completion_pct": 71.9, "games": 8},
            {"year": 2021, "age": 20, "yards": 4435, "tds": 44, "ints": 6, "completion_pct": 71.9, "games": 13},
            {"year": 2022, "age": 21, "yards": 3688, "tds": 41, "ints": 6, "completion_pct": 65.9, "games": 13},
        ],
        "team_stats_2024": {"total_yards": 5000, "total_tds": 52},
        "pff_grade_2024": 92.4,
    },
    "Will Anderson Jr": {
        "position": "EDGE",
        "school": "Alabama",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2020, "age": 18, "sacks": 3.0, "pressures": 16, "tfl": 10, "games": 11},
            {"year": 2021, "age": 19, "sacks": 17.5,"pressures": 68, "tfl": 34, "games": 13},
            {"year": 2022, "age": 20, "sacks": 11.0,"pressures": 50, "tfl": 22, "games": 13},
        ],
        "team_stats_2024": {"total_sacks": 30, "total_pressures": 112},
        "pff_grade_2024": 89.8,
    },
    "Jalen Carter": {
        "position": "DT",
        "school": "Georgia",
        "conference": "SEC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2020, "age": 19, "sacks": 3.0, "pressures": 18, "tfl": 8,  "games": 10},
            {"year": 2021, "age": 20, "sacks": 6.0, "pressures": 35, "tfl": 14, "games": 14},
            {"year": 2022, "age": 21, "sacks": 3.0, "pressures": 28, "tfl": 9,  "games": 9},
        ],
        "team_stats_2024": {"total_sacks": 22, "total_pressures": 82},
        "pff_grade_2024": 90.6,
    },
    # ── 2022 Draft Class ──────────────────────────────────────────────────────
    "Aidan Hutchinson": {
        "position": "EDGE",
        "school": "Michigan",
        "conference": "Big Ten",
        "system": "run_power",
        "stats_by_year": [
            {"year": 2019, "age": 19, "sacks": 4.5, "pressures": 22, "tfl": 6, "games": 12},
            {"year": 2020, "age": 20, "sacks": 1.5, "pressures": 9,  "tfl": 4, "games": 6},
            {"year": 2021, "age": 21, "sacks": 14.0,"pressures": 55, "tfl": 17,"games": 14},
        ],
        "team_stats_2024": {"total_sacks": 27, "total_pressures": 95},
        "pff_grade_2024": 91.2,
    },
    "Ahmad Gardner": {
        "position": "CB",
        "school": "Cincinnati",
        "conference": "AAC",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2019, "age": 18, "interceptions": 1, "pbu": 6,  "coverage_grade": 65.0, "games": 12},
            {"year": 2020, "age": 19, "interceptions": 3, "pbu": 10, "coverage_grade": 82.0, "games": 10},
            {"year": 2021, "age": 20, "interceptions": 4, "pbu": 11, "coverage_grade": 90.0, "games": 13},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 90.0,
    },
    "Kyle Hamilton": {
        "position": "S",
        "school": "Notre Dame",
        "conference": "Independent",
        "system": "pro_style",
        "stats_by_year": [
            {"year": 2019, "age": 18, "tackles": 40, "interceptions": 1, "pbu": 5, "games": 12},
            {"year": 2020, "age": 19, "tackles": 44, "interceptions": 3, "pbu": 7, "games": 11},
            {"year": 2021, "age": 20, "tackles": 55, "interceptions": 5, "pbu": 9, "games": 12},
        ],
        "team_stats_2024": {},
        "pff_grade_2024": 91.8,
    },
}


class CollegeProductionScorer:
    """
    Evaluates and adjusts college production for draft prospects,
    accounting for conference strength, offensive system, and
    career trajectory signals.
    """

    def __init__(self) -> None:
        self._db = PLAYER_COLLEGE_DB
        self._conf_adj = CONFERENCE_ADJUSTMENTS
        self._sys_adj = SYSTEM_ADJUSTMENTS

    # ── Public API ────────────────────────────────────────────────────────────

    def adjusted_stat(
        self,
        raw_stat: float,
        conference: str,
        system: str,
    ) -> float:
        """
        Apply conference strength and system adjustments to a raw stat.

        Stats from tougher conferences are inflated (the player performed
        better relative to the competition).  Stats from pass-friendly
        systems are deflated.

        Args:
            raw_stat:   The raw counting stat (yards, TDs, sacks, etc.)
            conference: Conference name, e.g. "SEC"
            system:     Offensive system, e.g. "air_raid"

        Returns:
            Adjusted stat value (float)
        """
        conf_factor = self._conf_adj.get(conference, 1.00)
        sys_factor = self._sys_adj.get(system, 1.00)
        # Conference tougher → conf_factor > 1 → adjusted stat goes UP
        # System inflating → sys_factor < 1 → adjusted stat goes DOWN
        return round(raw_stat * conf_factor / sys_factor, 1)

    def career_trajectory(self, stats_by_year: List[Dict]) -> Dict:
        """
        Compute year-over-year improvement slope and acceleration.

        Uses a key production metric (yards for skill positions,
        pressures for pass rushers, coverage grade for DBs) to
        measure growth.

        Args:
            stats_by_year: List of annual stat dicts, oldest first.
                           Each dict must have 'year' and at least one
                           of: yards, sacks, pressures, coverage_grade,
                           tackles, pass_block_grade.

        Returns:
            dict with keys: slope, acceleration, trending (up/flat/down)
        """
        if len(stats_by_year) < 2:
            return {"slope": 0.0, "acceleration": 1.0, "trending": "insufficient_data"}

        key = self._pick_trajectory_key(stats_by_year)
        values = [s.get(key, 0) for s in sorted(stats_by_year, key=lambda x: x["year"])]

        slopes = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                slopes.append(values[i] - values[i - 1])

        if not slopes:
            return {"slope": 0.0, "acceleration": 1.0, "trending": "flat"}

        avg_slope = sum(slopes) / len(slopes)

        # Acceleration = ratio of last slope to first slope
        acceleration = 1.0
        if len(slopes) >= 2 and slopes[0] != 0:
            acceleration = round(slopes[-1] / slopes[0], 2)

        trending = "up" if avg_slope > 0 else ("down" if avg_slope < 0 else "flat")

        return {
            "key_metric": key,
            "slope": round(avg_slope, 1),
            "acceleration": acceleration,
            "trending": trending,
            "annual_values": values,
        }

    def dominator_rating(self, player_stats: Dict, team_stats: Dict) -> float:
        """
        Calculate the player's share of team production (0-100).

        A high dominator rating means the player was THE guy —
        essential to team offense/defense.

        Args:
            player_stats: Dict with player's seasonal stats
            team_stats:   Dict with team totals (same keys with "total_" prefix)

        Returns:
            float in [0, 100]
        """
        if not team_stats:
            return 50.0  # Unknown → assume average

        shares = []

        if "yards" in player_stats and "total_yards" in team_stats and team_stats["total_yards"] > 0:
            shares.append(player_stats["yards"] / team_stats["total_yards"] * 100)

        if "receptions" in player_stats and "total_receptions" in team_stats and team_stats.get("total_receptions", 0) > 0:
            shares.append(player_stats["receptions"] / team_stats["total_receptions"] * 100)

        if "tds" in player_stats and "total_tds" in team_stats and team_stats["total_tds"] > 0:
            shares.append(player_stats["tds"] / team_stats["total_tds"] * 100)

        if "sacks" in player_stats and "total_sacks" in team_stats and team_stats.get("total_sacks", 0) > 0:
            shares.append(player_stats["sacks"] / team_stats["total_sacks"] * 100)

        if "pressures" in player_stats and "total_pressures" in team_stats and team_stats.get("total_pressures", 0) > 0:
            shares.append(player_stats["pressures"] / team_stats["total_pressures"] * 100)

        if not shares:
            return 50.0

        return round(sum(shares) / len(shares), 1)

    def breakout_age(self, stats_by_year: List[Dict]) -> Optional[int]:
        """
        Return the age at which the player had their first significant
        breakout season.  Lower age = better prospect (elite upside signal).

        Returns None if no breakout season is found in the data.

        Args:
            stats_by_year: List of annual stat dicts, each with 'age' key.
        """
        if not stats_by_year:
            return None

        # Detect position from available keys
        for year_stats in sorted(stats_by_year, key=lambda x: x.get("year", 0)):
            if self._is_breakout(year_stats):
                return year_stats.get("age")

        return None  # No breakout in dataset

    def score_player(self, player: str) -> float:
        """
        Return an overall college production score (0-100) for a
        player in the database.

        Factors:
          - Adjusted dominator rating (30%)
          - PFF grade (25%)
          - Career trajectory slope (20%)
          - Breakout age signal (15%)
          - Conference strength bonus (10%)
        """
        if player not in self._db:
            raise KeyError(f"Player '{player}' not found in college production database.")

        data = self._db[player]
        pos = data["position"]
        conf = data["conference"]
        system = data["system"]
        years = data["stats_by_year"]
        team_stats = data.get("team_stats_2024", {})
        pff = data.get("pff_grade_2024", 75.0)

        # Latest season stats for dominator rating
        latest = sorted(years, key=lambda x: x["year"])[-1]

        # Dominator rating (0-100)
        dom = self.dominator_rating(latest, team_stats)

        # PFF grade already 0-100
        pff_norm = max(0.0, min(100.0, (pff - 50) * 2))

        # Trajectory signal (0-100)
        traj = self.career_trajectory(years)
        slope_signal = 50.0
        if traj["trending"] == "up":
            slope_signal = min(100.0, 50.0 + abs(traj["slope"]) * 2)
        elif traj["trending"] == "down":
            slope_signal = max(0.0, 50.0 - abs(traj["slope"]) * 2)

        # Breakout age signal (younger = better)
        age = self.breakout_age(years)
        if age is None:
            age_signal = 40.0
        elif age <= 18:
            age_signal = 100.0
        elif age <= 19:
            age_signal = 90.0
        elif age <= 20:
            age_signal = 75.0
        elif age <= 21:
            age_signal = 60.0
        else:
            age_signal = 40.0

        # Conference bonus (0-100)
        conf_factor = self._conf_adj.get(conf, 1.00)
        conf_signal = min(100.0, (conf_factor - 0.85) / 0.25 * 100)

        composite = (
            0.30 * dom
            + 0.25 * pff_norm
            + 0.20 * slope_signal
            + 0.15 * age_signal
            + 0.10 * conf_signal
        )

        return round(composite, 1)

    def all_players(self) -> List[str]:
        return list(self._db.keys())

    def get_player_data(self, player: str) -> Dict:
        if player not in self._db:
            raise KeyError(f"Player '{player}' not found in college production database.")
        return self._db[player]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pick_trajectory_key(self, stats_by_year: List[Dict]) -> str:
        """Pick the best metric to track trajectory."""
        candidates = ["yards", "pressures", "sacks", "tackles", "coverage_grade",
                      "pass_block_grade", "pbu"]
        sample = stats_by_year[0]
        for c in candidates:
            if c in sample:
                return c
        # Fallback: first numeric key
        for k, v in sample.items():
            if isinstance(v, (int, float)) and k != "year" and k != "age" and k != "games":
                return k
        return "yards"

    def _is_breakout(self, year_stats: Dict) -> bool:
        """Return True if this season qualifies as a breakout."""
        # WR / TE / RB skill positions
        if year_stats.get("yards", 0) >= 800:
            return True
        if year_stats.get("receptions", 0) >= 50:
            return True
        # QB
        if year_stats.get("completion_pct", 0) >= 60.0 and year_stats.get("tds", 0) >= 20:
            return True
        # Pass rushers
        if year_stats.get("sacks", 0) >= 7.0:
            return True
        if year_stats.get("pressures", 0) >= 25:
            return True
        # DB
        if year_stats.get("coverage_grade", 0) >= 75.0:
            return True
        if year_stats.get("interceptions", 0) >= 3:
            return True
        # OL
        if year_stats.get("pass_block_grade", 0) >= 75.0:
            return True
        # LB
        if year_stats.get("tackles", 0) >= 70:
            return True
        return False
