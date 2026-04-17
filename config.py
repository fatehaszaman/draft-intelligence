"""
config.py — Global configuration for draft-intelligence.

Adjust these values to tune model behavior, enable live data feeds,
and control output verbosity.
"""

# ── Data source ───────────────────────────────────────────────────────────────
# Set True to pull live combine/draft data via nfl-data-py.
# Set False to use the hardcoded fallback data (no network required).
USE_LIVE_DATA: bool = False

# ── Composite grade weights ───────────────────────────────────────────────────
# Must sum to 1.0
COMBINE_WEIGHT: float = 0.25
COLLEGE_WEIGHT: float = 0.30
COMPS_WEIGHT: float = 0.25
INTANGIBLES_WEIGHT: float = 0.20

assert abs(
    COMBINE_WEIGHT + COLLEGE_WEIGHT + COMPS_WEIGHT + INTANGIBLES_WEIGHT - 1.0
) < 1e-9, "Grade weights must sum to 1.0"

# ── Psychological vs Physical weighting ───────────────────────────────────────
# Used in two places:
#   1. readiness_composite = (psych × PSYCH_WEIGHT + physical × PHYSICAL_WEIGHT) / TOTAL_WEIGHT
#   2. overall_grade       = (technical × PHYSICAL_WEIGHT + psych × PSYCH_WEIGHT) / TOTAL_WEIGHT
#
# Weighting: physical/technical (1.5) outweighs psychological (1.0) on 2.5 total scale.
# Rationale: NFL Draft evaluation is primarily talent-driven. A physically elite
# prospect with moderate psychological concerns (score 65/100) still projects as
# a quality starter. However the 1.0 psych weight is non-trivial — it's the
# difference between JaMarcus Russell (psych: 35) and Patrick Mahomes (psych: 91)
# at similar physical grades. Known busts almost always show psych composite < 60.
PSYCH_WEIGHT:    float = 1.0
PHYSICAL_WEIGHT: float = 1.5
TOTAL_WEIGHT:    float = PSYCH_WEIGHT + PHYSICAL_WEIGHT   # 2.5

# ── Career success model weights ──────────────────────────────────────────────
CAREER_COMPS_WEIGHT: float = 0.40
CAREER_ATHLETICISM_WEIGHT: float = 0.25
CAREER_PRODUCTION_WEIGHT: float = 0.25
CAREER_INTANGIBLES_WEIGHT: float = 0.10

# ── Mock draft settings ───────────────────────────────────────────────────────
ROUNDS_TO_SIMULATE: int = 3
ENABLE_TRADES: bool = False          # Trade logic not yet implemented
TOP_COMPS_TO_RETURN: int = 3         # KNN neighbor count for historical comps

# ── Conference strength multipliers ──────────────────────────────────────────
# Values > 1.0 → reduce raw stats (competition was tougher)
# Values < 1.0 → boost raw stats (competition was weaker)
CONFERENCE_ADJUSTMENTS: dict = {
    "SEC": 1.06,
    "Big Ten": 1.04,
    "ACC": 1.02,
    "Big 12": 1.01,
    "PAC-12": 1.00,
    "AAC": 0.95,
    "Mountain West": 0.93,
    "Sun Belt": 0.91,
    "MAC": 0.90,
    "CUSA": 0.89,
    "Independent": 0.97,
}

# ── Offensive system multipliers ─────────────────────────────────────────────
# Air raid inflates counting stats — apply penalty
SYSTEM_ADJUSTMENTS: dict = {
    "air_raid": 0.88,
    "spread": 0.93,
    "pro_style": 1.00,
    "west_coast": 0.97,
    "run_power": 1.00,
    "pistol": 0.95,
}

# ── Output ────────────────────────────────────────────────────────────────────
VERBOSE: bool = True
RICH_OUTPUT: bool = True             # Use Rich library for pretty tables
LOG_LEVEL: str = "INFO"              # DEBUG | INFO | WARNING | ERROR

# ── Paths ─────────────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_PROSPECTS_PATH = os.path.join(DATA_DIR, "sample_prospects.json")
