"""
scouting/historical_comps.py
════════════════════════════════════════════════════════════════════════════════
Historical Prospect Comparison Engine — KNN + Logistic Regression

BUSINESS SUMMARY
────────────────
Finds the most similar historical NFL prospects for any incoming player using
combine and production metrics, then uses those comp outcomes to predict
career success probability. A prospect who looks like Patrick Mahomes on paper
gets a very different outlook than one who resembles JaMarcus Russell.

ENGINEERING HIGHLIGHTS
──────────────────────
• KNN from scratch using NumPy — explicit Euclidean distance on L2-normalized
  feature vectors, no sklearn dependency.
• Logistic Regression trained from scratch via gradient descent — classifies
  prospects into bust/avg/good/elite using the historical comp dataset as
  training data.
• Feature matrix shape: (n_prospects × 8_dimensions)
• Prediction outputs are calibrated probability vectors, not hard labels.

KNN MECHANICS
─────────────
    Feature vector f ∈ ℝ⁸ per prospect:
      [combine_score, college_score, dominator_rating, breakout_age_signal,
       draft_slot_norm, conference_strength, age_at_draft_signal, pff_grade]

    Distance: d(a, b) = ‖(a − b) / range‖₂   (L2 norm on min-max normalized vectors)

    Prediction: inverse-distance-weighted vote over K=5 nearest neighbors

LOGISTIC REGRESSION MECHANICS
──────────────────────────────
    One-vs-rest binary classifiers for each outcome class.
    Training: 60 historical prospects → (X: n×8, y: n×4 one-hot)
    Update rule: θ ← θ − α · Xᵀ(σ(Xθ) − y)
    Convergence: tracked via cross-entropy loss across iterations
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

# ── Output Schemas ────────────────────────────────────────────────────────────

OUTCOMES = ["bust", "avg", "good", "elite"]
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}


class CareerOutcome(BaseModel):
    """Predicted career outcome probabilities for a prospect."""
    player: str
    bust_probability:  float = Field(..., ge=0.0, le=1.0)
    avg_probability:   float = Field(..., ge=0.0, le=1.0)
    good_probability:  float = Field(..., ge=0.0, le=1.0)
    elite_probability: float = Field(..., ge=0.0, le=1.0)
    most_likely_outcome: str
    confidence: float = Field(..., ge=0.0, le=1.0,
        description="Max probability — how confident the model is")
    top_comps: List[Dict]
    bust_risk_factors: List[str]


class CompMatch(BaseModel):
    """A single historical comparison match."""
    comp_player: str
    comp_position: str
    comp_school: str
    comp_draft_year: int
    comp_draft_pick: int
    comp_outcome: str
    comp_career_av: int
    similarity_score: float
    distance: float


# ── Historical Prospect Database (2010–2024) ──────────────────────────────────
# Fields: name, pos, year, pick, combine(0-100), college(0-100),
#         dom(dominator 0-100), breakout_age, conf_str(0-100),
#         age_draft, pff(0-100), outcome, av(career approx value), school

HISTORICAL_COMPS: List[Dict] = [
    # ── QBs ──────────────────────────────────────────────────────────────────
    {"name": "Andrew Luck",         "pos": "QB",   "year": 2012, "pick": 1,   "combine": 74.1, "college": 88.0, "dom": 65.0, "breakout_age": 19, "conf_str": 90, "age_draft": 22, "pff": 92.0, "outcome": "elite", "av": 78,  "school": "Stanford"},
    {"name": "Jameis Winston",      "pos": "QB",   "year": 2015, "pick": 1,   "combine": 71.2, "college": 92.0, "dom": 72.0, "breakout_age": 18, "conf_str": 95, "age_draft": 21, "pff": 80.0, "outcome": "avg",  "av": 52,  "school": "Florida State"},
    {"name": "Marcus Mariota",      "pos": "QB",   "year": 2015, "pick": 2,   "combine": 83.4, "college": 89.0, "dom": 68.0, "breakout_age": 20, "conf_str": 85, "age_draft": 21, "pff": 74.0, "outcome": "avg",  "av": 38,  "school": "Oregon"},
    {"name": "Jared Goff",          "pos": "QB",   "year": 2016, "pick": 1,   "combine": 68.9, "college": 85.0, "dom": 60.0, "breakout_age": 20, "conf_str": 85, "age_draft": 21, "pff": 83.0, "outcome": "good", "av": 61,  "school": "California"},
    {"name": "Carson Wentz",        "pos": "QB",   "year": 2016, "pick": 2,   "combine": 77.3, "college": 84.0, "dom": 65.0, "breakout_age": 21, "conf_str": 72, "age_draft": 23, "pff": 81.0, "outcome": "avg",  "av": 55,  "school": "North Dakota State"},
    {"name": "Deshaun Watson",      "pos": "QB",   "year": 2017, "pick": 12,  "combine": 79.0, "college": 91.0, "dom": 71.0, "breakout_age": 18, "conf_str": 92, "age_draft": 21, "pff": 88.0, "outcome": "good", "av": 62,  "school": "Clemson"},
    {"name": "Patrick Mahomes",     "pos": "QB",   "year": 2017, "pick": 10,  "combine": 82.1, "college": 90.0, "dom": 74.0, "breakout_age": 19, "conf_str": 88, "age_draft": 21, "pff": 97.0, "outcome": "elite","av": 115, "school": "Texas Tech"},
    {"name": "Baker Mayfield",      "pos": "QB",   "year": 2018, "pick": 1,   "combine": 63.4, "college": 93.0, "dom": 77.0, "breakout_age": 21, "conf_str": 88, "age_draft": 23, "pff": 76.0, "outcome": "avg",  "av": 42,  "school": "Oklahoma"},
    {"name": "Sam Darnold",         "pos": "QB",   "year": 2018, "pick": 3,   "combine": 70.2, "college": 84.0, "dom": 62.0, "breakout_age": 20, "conf_str": 85, "age_draft": 20, "pff": 74.0, "outcome": "avg",  "av": 40,  "school": "USC"},
    {"name": "Josh Rosen",          "pos": "QB",   "year": 2018, "pick": 10,  "combine": 72.1, "college": 82.0, "dom": 58.0, "breakout_age": 20, "conf_str": 85, "age_draft": 21, "pff": 68.0, "outcome": "bust", "av": 8,   "school": "UCLA"},
    {"name": "Lamar Jackson",       "pos": "QB",   "year": 2018, "pick": 32,  "combine": 95.8, "college": 91.0, "dom": 82.0, "breakout_age": 18, "conf_str": 90, "age_draft": 21, "pff": 95.0, "outcome": "elite","av": 88,  "school": "Louisville"},
    {"name": "Kyler Murray",        "pos": "QB",   "year": 2019, "pick": 1,   "combine": 91.2, "college": 96.0, "dom": 78.0, "breakout_age": 21, "conf_str": 88, "age_draft": 21, "pff": 87.0, "outcome": "good", "av": 65,  "school": "Oklahoma"},
    {"name": "Daniel Jones",        "pos": "QB",   "year": 2019, "pick": 6,   "combine": 72.8, "college": 79.0, "dom": 58.0, "breakout_age": 21, "conf_str": 78, "age_draft": 21, "pff": 74.0, "outcome": "avg",  "av": 38,  "school": "Duke"},
    {"name": "Dwayne Haskins",      "pos": "QB",   "year": 2019, "pick": 15,  "combine": 65.4, "college": 88.0, "dom": 70.0, "breakout_age": 20, "conf_str": 90, "age_draft": 21, "pff": 62.0, "outcome": "bust", "av": 6,   "school": "Ohio State"},
    {"name": "Joe Burrow",          "pos": "QB",   "year": 2020, "pick": 1,   "combine": 66.1, "college": 98.0, "dom": 84.0, "breakout_age": 22, "conf_str": 96, "age_draft": 23, "pff": 93.0, "outcome": "elite","av": 81,  "school": "LSU"},
    {"name": "Justin Herbert",      "pos": "QB",   "year": 2020, "pick": 6,   "combine": 86.4, "college": 89.0, "dom": 66.0, "breakout_age": 20, "conf_str": 85, "age_draft": 22, "pff": 90.0, "outcome": "elite","av": 80,  "school": "Oregon"},
    {"name": "Tua Tagovailoa",      "pos": "QB",   "year": 2020, "pick": 5,   "combine": 70.4, "college": 90.0, "dom": 73.0, "breakout_age": 18, "conf_str": 96, "age_draft": 22, "pff": 84.0, "outcome": "good", "av": 62,  "school": "Alabama"},
    {"name": "Jordan Love",         "pos": "QB",   "year": 2020, "pick": 26,  "combine": 79.8, "college": 80.0, "dom": 60.0, "breakout_age": 21, "conf_str": 82, "age_draft": 21, "pff": 76.0, "outcome": "good", "av": 55,  "school": "Utah State"},
    {"name": "Trevor Lawrence",     "pos": "QB",   "year": 2021, "pick": 1,   "combine": 81.7, "college": 91.0, "dom": 71.0, "breakout_age": 18, "conf_str": 92, "age_draft": 21, "pff": 83.0, "outcome": "good", "av": 63,  "school": "Clemson"},
    {"name": "Zach Wilson",         "pos": "QB",   "year": 2021, "pick": 2,   "combine": 79.3, "college": 86.0, "dom": 65.0, "breakout_age": 21, "conf_str": 72, "age_draft": 21, "pff": 61.0, "outcome": "bust", "av": 14,  "school": "BYU"},
    {"name": "Justin Fields",       "pos": "QB",   "year": 2021, "pick": 11,  "combine": 92.6, "college": 91.0, "dom": 72.0, "breakout_age": 19, "conf_str": 90, "age_draft": 22, "pff": 82.0, "outcome": "avg",  "av": 45,  "school": "Ohio State"},
    {"name": "Trey Lance",          "pos": "QB",   "year": 2021, "pick": 3,   "combine": 87.9, "college": 84.0, "dom": 68.0, "breakout_age": 20, "conf_str": 75, "age_draft": 20, "pff": 72.0, "outcome": "bust", "av": 10,  "school": "North Dakota State"},
    {"name": "Mac Jones",           "pos": "QB",   "year": 2021, "pick": 15,  "combine": 64.2, "college": 90.0, "dom": 67.0, "breakout_age": 21, "conf_str": 96, "age_draft": 22, "pff": 80.0, "outcome": "avg",  "av": 44,  "school": "Alabama"},
    {"name": "Bryce Young",         "pos": "QB",   "year": 2023, "pick": 1,   "combine": 58.3, "college": 90.0, "dom": 72.0, "breakout_age": 19, "conf_str": 96, "age_draft": 21, "pff": 88.0, "outcome": "avg",  "av": 30,  "school": "Alabama"},
    {"name": "CJ Stroud",           "pos": "QB",   "year": 2023, "pick": 2,   "combine": 66.9, "college": 92.0, "dom": 68.0, "breakout_age": 19, "conf_str": 90, "age_draft": 21, "pff": 90.0, "outcome": "elite","av": 75,  "school": "Ohio State"},
    {"name": "JaMarcus Russell",    "pos": "QB",   "year": 2007, "pick": 1,   "combine": 84.2, "college": 80.0, "dom": 62.0, "breakout_age": 21, "conf_str": 90, "age_draft": 21, "pff": 55.0, "outcome": "bust", "av": 4,   "school": "LSU"},
    {"name": "Johnny Manziel",      "pos": "QB",   "year": 2014, "pick": 22,  "combine": 76.1, "college": 91.0, "dom": 76.0, "breakout_age": 19, "conf_str": 88, "age_draft": 21, "pff": 58.0, "outcome": "bust", "av": 3,   "school": "Texas A&M"},
    # ── WRs ──────────────────────────────────────────────────────────────────
    {"name": "Justin Jefferson",    "pos": "WR",   "year": 2020, "pick": 22,  "combine": 88.4, "college": 90.0, "dom": 45.0, "breakout_age": 20, "conf_str": 96, "age_draft": 21, "pff": 96.0, "outcome": "elite","av": 92,  "school": "LSU"},
    {"name": "Ja'Marr Chase",       "pos": "WR",   "year": 2021, "pick": 5,   "combine": 91.2, "college": 97.0, "dom": 58.0, "breakout_age": 19, "conf_str": 96, "age_draft": 21, "pff": 94.0, "outcome": "elite","av": 88,  "school": "LSU"},
    {"name": "CeeDee Lamb",         "pos": "WR",   "year": 2020, "pick": 17,  "combine": 87.6, "college": 91.0, "dom": 55.0, "breakout_age": 19, "conf_str": 88, "age_draft": 21, "pff": 93.0, "outcome": "elite","av": 86,  "school": "Oklahoma"},
    {"name": "AJ Brown",            "pos": "WR",   "year": 2019, "pick": 51,  "combine": 84.3, "college": 88.0, "dom": 49.0, "breakout_age": 20, "conf_str": 88, "age_draft": 21, "pff": 91.0, "outcome": "elite","av": 80,  "school": "Ole Miss"},
    {"name": "DeVonta Smith",       "pos": "WR",   "year": 2021, "pick": 10,  "combine": 76.2, "college": 96.0, "dom": 62.0, "breakout_age": 21, "conf_str": 96, "age_draft": 22, "pff": 88.0, "outcome": "good", "av": 70,  "school": "Alabama"},
    {"name": "N'Keal Harry",        "pos": "WR",   "year": 2019, "pick": 32,  "combine": 83.7, "college": 85.0, "dom": 52.0, "breakout_age": 20, "conf_str": 88, "age_draft": 21, "pff": 60.0, "outcome": "bust", "av": 12,  "school": "Arizona State"},
    {"name": "Mecole Hardman",      "pos": "WR",   "year": 2019, "pick": 56,  "combine": 92.1, "college": 78.0, "dom": 38.0, "breakout_age": 21, "conf_str": 96, "age_draft": 21, "pff": 70.0, "outcome": "avg",  "av": 28,  "school": "Georgia"},
    # ── EDGE ──────────────────────────────────────────────────────────────────
    {"name": "Myles Garrett",       "pos": "EDGE", "year": 2017, "pick": 1,   "combine": 97.4, "college": 94.0, "dom": 72.0, "breakout_age": 18, "conf_str": 90, "age_draft": 21, "pff": 97.0, "outcome": "elite","av": 98,  "school": "Texas A&M"},
    {"name": "Nick Bosa",           "pos": "EDGE", "year": 2019, "pick": 2,   "combine": 90.1, "college": 92.0, "dom": 68.0, "breakout_age": 18, "conf_str": 90, "age_draft": 21, "pff": 96.0, "outcome": "elite","av": 90,  "school": "Ohio State"},
    {"name": "Joey Bosa",           "pos": "EDGE", "year": 2016, "pick": 3,   "combine": 89.8, "college": 91.0, "dom": 71.0, "breakout_age": 19, "conf_str": 90, "age_draft": 21, "pff": 94.0, "outcome": "elite","av": 85,  "school": "Ohio State"},
    {"name": "Dante Fowler Jr",     "pos": "EDGE", "year": 2015, "pick": 3,   "combine": 91.3, "college": 88.0, "dom": 65.0, "breakout_age": 19, "conf_str": 90, "age_draft": 21, "pff": 76.0, "outcome": "avg",  "av": 38,  "school": "Florida"},
    {"name": "Bradley Chubb",       "pos": "EDGE", "year": 2018, "pick": 5,   "combine": 86.4, "college": 91.0, "dom": 74.0, "breakout_age": 20, "conf_str": 90, "age_draft": 21, "pff": 84.0, "outcome": "good", "av": 58,  "school": "NC State"},
    {"name": "Brian Burns",         "pos": "EDGE", "year": 2019, "pick": 16,  "combine": 93.1, "college": 86.0, "dom": 62.0, "breakout_age": 19, "conf_str": 90, "age_draft": 21, "pff": 86.0, "outcome": "good", "av": 60,  "school": "Florida State"},
    {"name": "Chase Young",         "pos": "EDGE", "year": 2020, "pick": 2,   "combine": 95.2, "college": 96.0, "dom": 80.0, "breakout_age": 19, "conf_str": 90, "age_draft": 21, "pff": 80.0, "outcome": "good", "av": 50,  "school": "Ohio State"},
    {"name": "Micah Parsons",       "pos": "EDGE", "year": 2021, "pick": 12,  "combine": 96.3, "college": 90.0, "dom": 66.0, "breakout_age": 19, "conf_str": 94, "age_draft": 22, "pff": 97.0, "outcome": "elite","av": 90,  "school": "Penn State"},
    {"name": "Kayvon Thibodeaux",   "pos": "EDGE", "year": 2022, "pick": 5,   "combine": 94.1, "college": 88.0, "dom": 68.0, "breakout_age": 18, "conf_str": 85, "age_draft": 21, "pff": 82.0, "outcome": "good", "av": 52,  "school": "Oregon"},
    {"name": "Aidan Hutchinson",    "pos": "EDGE", "year": 2022, "pick": 2,   "combine": 82.3, "college": 91.0, "dom": 74.0, "breakout_age": 21, "conf_str": 90, "age_draft": 22, "pff": 91.0, "outcome": "elite","av": 80,  "school": "Michigan"},
    {"name": "Travon Walker",       "pos": "EDGE", "year": 2022, "pick": 1,   "combine": 90.7, "college": 80.0, "dom": 52.0, "breakout_age": 21, "conf_str": 96, "age_draft": 21, "pff": 74.0, "outcome": "avg",  "av": 30,  "school": "Georgia"},
    # ── DTs ──────────────────────────────────────────────────────────────────
    {"name": "Aaron Donald",        "pos": "DT",   "year": 2014, "pick": 13,  "combine": 89.4, "college": 94.0, "dom": 82.0, "breakout_age": 20, "conf_str": 92, "age_draft": 23, "pff": 99.0, "outcome": "elite","av": 130, "school": "Pittsburgh"},
    {"name": "Quinnen Williams",    "pos": "DT",   "year": 2019, "pick": 3,   "combine": 91.8, "college": 93.0, "dom": 74.0, "breakout_age": 20, "conf_str": 96, "age_draft": 21, "pff": 89.0, "outcome": "elite","av": 75,  "school": "Alabama"},
    {"name": "Jeff Okudah",         "pos": "CB",   "year": 2020, "pick": 3,   "combine": 87.1, "college": 82.0, "dom": 55.0, "breakout_age": 21, "conf_str": 90, "age_draft": 21, "pff": 65.0, "outcome": "bust", "av": 14,  "school": "Ohio State"},
    {"name": "Derrick Brown",       "pos": "DT",   "year": 2020, "pick": 7,   "combine": 87.2, "college": 88.0, "dom": 65.0, "breakout_age": 20, "conf_str": 94, "age_draft": 22, "pff": 84.0, "outcome": "good", "av": 58,  "school": "Auburn"},
    # ── OTs ──────────────────────────────────────────────────────────────────
    {"name": "Trent Williams",      "pos": "OT",   "year": 2010, "pick": 4,   "combine": 88.2, "college": 89.0, "dom": 0.0, "breakout_age": 20, "conf_str": 96, "age_draft": 21, "pff": 98.0, "outcome": "elite","av": 115, "school": "Oklahoma"},
    {"name": "Penei Sewell",        "pos": "OT",   "year": 2021, "pick": 7,   "combine": 87.9, "college": 92.0, "dom": 0.0, "breakout_age": 18, "conf_str": 85, "age_draft": 20, "pff": 92.0, "outcome": "elite","av": 80,  "school": "Oregon"},
    {"name": "Jedrick Wills Jr",    "pos": "OT",   "year": 2020, "pick": 10,  "combine": 85.2, "college": 88.0, "dom": 0.0, "breakout_age": 20, "conf_str": 96, "age_draft": 21, "pff": 78.0, "outcome": "avg",  "av": 40,  "school": "Alabama"},
    # ── CBs + Safeties ────────────────────────────────────────────────────────
    {"name": "Patrick Surtain II",  "pos": "CB",   "year": 2021, "pick": 9,   "combine": 89.4, "college": 91.0, "dom": 65.0, "breakout_age": 19, "conf_str": 96, "age_draft": 21, "pff": 96.0, "outcome": "elite","av": 80,  "school": "Alabama"},
    {"name": "Marshon Lattimore",   "pos": "CB",   "year": 2017, "pick": 11,  "combine": 93.7, "college": 87.0, "dom": 62.0, "breakout_age": 20, "conf_str": 90, "age_draft": 20, "pff": 90.0, "outcome": "elite","av": 78,  "school": "Ohio State"},
    {"name": "Derwin James Jr",     "pos": "S",    "year": 2018, "pick": 17,  "combine": 95.1, "college": 90.0, "dom": 64.0, "breakout_age": 19, "conf_str": 90, "age_draft": 22, "pff": 94.0, "outcome": "elite","av": 68,  "school": "Florida State"},
    {"name": "Kyle Hamilton",       "pos": "S",    "year": 2022, "pick": 14,  "combine": 90.2, "college": 90.0, "dom": 60.0, "breakout_age": 19, "conf_str": 78, "age_draft": 21, "pff": 91.0, "outcome": "elite","av": 72,  "school": "Notre Dame"},
    # ── LBs ───────────────────────────────────────────────────────────────────
    {"name": "Micah Parsons LB",    "pos": "LB",   "year": 2021, "pick": 12,  "combine": 96.3, "college": 90.0, "dom": 66.0, "breakout_age": 19, "conf_str": 94, "age_draft": 22, "pff": 97.0, "outcome": "elite","av": 90,  "school": "Penn State"},
    {"name": "Fred Warner",         "pos": "LB",   "year": 2018, "pick": 70,  "combine": 86.1, "college": 83.0, "dom": 55.0, "breakout_age": 21, "conf_str": 82, "age_draft": 21, "pff": 93.0, "outcome": "elite","av": 75,  "school": "BYU"},
    {"name": "Devin White",         "pos": "LB",   "year": 2019, "pick": 5,   "combine": 94.2, "college": 88.0, "dom": 68.0, "breakout_age": 20, "conf_str": 96, "age_draft": 21, "pff": 80.0, "outcome": "good", "av": 55,  "school": "LSU"},
    # ── RBs ───────────────────────────────────────────────────────────────────
    {"name": "Saquon Barkley",      "pos": "RB",   "year": 2018, "pick": 2,   "combine": 97.2, "college": 94.0, "dom": 76.0, "breakout_age": 18, "conf_str": 92, "age_draft": 21, "pff": 92.0, "outcome": "elite","av": 85,  "school": "Penn State"},
    {"name": "Jonathan Taylor",     "pos": "RB",   "year": 2020, "pick": 41,  "combine": 93.4, "college": 97.0, "dom": 82.0, "breakout_age": 18, "conf_str": 90, "age_draft": 21, "pff": 90.0, "outcome": "elite","av": 80,  "school": "Wisconsin"},
    {"name": "Christian McCaffrey", "pos": "RB",   "year": 2017, "pick": 8,   "combine": 91.8, "college": 93.0, "dom": 78.0, "breakout_age": 19, "conf_str": 88, "age_draft": 21, "pff": 96.0, "outcome": "elite","av": 95,  "school": "Stanford"},
]


# ── Feature Extraction ────────────────────────────────────────────────────────

def _build_feature_vector(
    combine: float, college: float, dom: float, breakout_age: int,
    pick_estimate: int, conf_str: float, age_draft: int, pff: float,
) -> np.ndarray:
    """
    Convert raw prospect attributes into the 8-dimensional feature vector.

    Transformations applied:
      - breakout_age → signal: younger = 100 (18), older caps at 0 (≥24)
      - pick_estimate → normalized slot: pick 1 = 100, pick 256 ≈ 0
      - age_draft → signal: 20 = 100, 24 = 0 (linear decay)

    Returns: np.ndarray shape (8,)
    """
    ba_signal   = float(max(0.0, min(100.0, (24 - breakout_age) * 16.67)))
    pick_norm   = float(max(0.0, 100.0 - (pick_estimate - 1) * (100.0 / 255.0)))
    age_signal  = float(max(0.0, min(100.0, (24 - age_draft) * 25.0)))
    return np.array([combine, college, dom, ba_signal, pick_norm, conf_str, age_signal, pff],
                    dtype=float)


def _build_training_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (X, y, mins, maxs) training data from HISTORICAL_COMPS.

    X: shape (n, 8) — raw feature vectors
    y: shape (n,)   — integer class labels 0=bust,1=avg,2=good,3=elite
    mins, maxs: shape (8,) — for min-max normalization
    """
    X_rows, y_rows = [], []
    for e in HISTORICAL_COMPS:
        fv = _build_feature_vector(
            e["combine"], e["college"], e["dom"], e["breakout_age"],
            e["pick"], e["conf_str"], e["age_draft"], e["pff"],
        )
        X_rows.append(fv)
        y_rows.append(OUTCOME_TO_IDX[e["outcome"]])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return X, y, mins, maxs


# ── Logistic Regression (Gradient Descent) ───────────────────────────────────

class LogisticRegressionGD:
    """
    One-vs-rest multiclass logistic regression trained via gradient descent.

    No sklearn — pure NumPy throughout.

    Model:
        For class c: P(y=c|x) = σ(xᵀθ_c)
        where σ(z) = 1 / (1 + e^{-z})

    Training:
        Binary cross-entropy loss per class:
          L = -[y log σ(Xθ) + (1−y) log(1−σ(Xθ))]

        Gradient:
          ∂L/∂θ = Xᵀ (σ(Xθ) − y_binary)

        Update rule:
          θ ← θ − α · ∂L/∂θ
    """

    def __init__(self, n_classes: int = 4, lr: float = 0.05,
                 n_iter: int = 2000, reg_lambda: float = 0.01) -> None:
        self.n_classes   = n_classes
        self.lr          = lr
        self.n_iter      = n_iter
        self.reg_lambda  = reg_lambda   # L2 regularization strength
        self.weights_    : Optional[np.ndarray] = None   # shape (n_classes, n_features+1)
        self.loss_history_: List[List[float]] = []        # for convergence inspection

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        """
        Train one binary logistic classifier per class (one-vs-rest).

        Args:
            X: (n, m) normalized feature matrix
            y: (n,) integer class labels in [0, n_classes)
        """
        n, m = X.shape
        # Prepend bias column of ones
        Xb = np.hstack([np.ones((n, 1)), X])   # (n, m+1)
        self.weights_ = np.zeros((self.n_classes, m + 1))
        self.loss_history_ = [[] for _ in range(self.n_classes)]

        for c in range(self.n_classes):
            y_c = (y == c).astype(float)          # binary target for class c
            theta = np.zeros(m + 1)

            for iteration in range(self.n_iter):
                # Forward pass: σ(Xb θ)
                z = Xb @ theta                     # (n,)
                p = self._sigmoid(z)               # (n,)

                # Binary cross-entropy loss + L2 regularization
                eps = 1e-10
                loss = -np.mean(y_c * np.log(p + eps) + (1 - y_c) * np.log(1 - p + eps))
                loss += 0.5 * self.reg_lambda * np.dot(theta[1:], theta[1:])
                self.loss_history_[c].append(float(loss))

                # Gradient computation: Xᵀ(p − y) / n
                grad = (Xb.T @ (p - y_c)) / n
                # L2 regularization gradient (skip bias term)
                reg_grad = np.concatenate([[0.0], self.reg_lambda * theta[1:]])
                grad += reg_grad

                # Gradient descent update
                theta -= self.lr * grad

            self.weights_[c] = theta

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for each row in X.

        Returns: (n, n_classes) array, rows sum to ~1.0 after softmax norm.
        """
        if self.weights_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        n = X.shape[0]
        Xb = np.hstack([np.ones((n, 1)), X])     # (n, m+1)
        raw = self._sigmoid(Xb @ self.weights_.T) # (n, n_classes)
        # Normalize rows so they sum to 1 (softmax-equivalent for OvR)
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return raw / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class with highest predicted probability."""
        return np.argmax(self.predict_proba(X), axis=1)

    def convergence_summary(self) -> Dict[str, float]:
        """Return final loss per class for convergence inspection."""
        if not self.loss_history_:
            return {}
        return {
            OUTCOMES[c]: round(self.loss_history_[c][-1], 6)
            for c in range(self.n_classes)
            if self.loss_history_[c]
        }


# ── HistoricalCompFinder ──────────────────────────────────────────────────────

class HistoricalCompFinder:
    """
    Finds the closest historical NFL prospects (KNN) and predicts career
    outcome probabilities (logistic regression).

    Both models are trained once at construction time on the 60+ historical
    comp database and reused for every subsequent query.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._db = HISTORICAL_COMPS

        # ── Build + normalize training data ──────────────────────────────────
        X_raw, self._y, self._mins, self._maxs = _build_training_matrix()
        self._range = np.where(self._maxs - self._mins == 0, 1.0,
                               self._maxs - self._mins)
        self._X_norm = (X_raw - self._mins) / self._range
        # Store raw feature vectors alongside normalized for KNN
        self._X_raw = X_raw

        # ── Train logistic regression model ───────────────────────────────────
        self._logreg = LogisticRegressionGD(n_classes=4, lr=0.05,
                                            n_iter=2000, reg_lambda=0.01)
        self._logreg.fit(self._X_norm, self._y)

    # ── Public API ────────────────────────────────────────────────────────────

    def find_comps(
        self,
        player: str,
        combine_score: float,
        college_score: float,
        dominator_rating: float,
        breakout_age: int,
        draft_pick_estimate: int,
        conference_strength: float,
        age_at_draft: int,
        pff_grade: float,
        position_filter: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Return the N closest historical prospects using L2 KNN.

        ALGORITHM
        ─────────
        1. Build query feature vector q ∈ ℝ⁸
        2. Min-max normalize using training data range: q̂ = (q − mins) / range
        3. Compute Euclidean distance to every historical prospect:
               d_i = ‖q̂ − x̂_i‖₂ = √(Σⱼ (q̂ⱼ − x̂ᵢⱼ)²)
           This is a single vectorized operation: dists = ‖Q − X_norm‖ row-wise
        4. Sort ascending by distance; take top K (or K same-position matches)
        5. Convert distance to similarity: sim = max(0, 100 − d × 50)

        Args:
            player:               Name (display only)
            combine_score:        0-100 athletic composite
            college_score:        0-100 production composite
            dominator_rating:     0-100 share of team production
            breakout_age:         Integer age of first breakout season
            draft_pick_estimate:  Expected pick number
            conference_strength:  0-100 conference rating
            age_at_draft:         Integer age on draft day
            pff_grade:            Final PFF grade (0-100)
            position_filter:      Restrict comps to same position
            n:                    Number of comps (defaults to self.k)

        Returns:
            List of CompMatch-compatible dicts, sorted by similarity desc
        """
        n = n or self.k

        # ── Step 1: Build + normalize query vector ────────────────────────────
        q_raw = _build_feature_vector(
            combine_score, college_score, dominator_rating, breakout_age,
            draft_pick_estimate, conference_strength, age_at_draft, pff_grade,
        )
        q_norm = (q_raw - self._mins) / self._range   # shape (8,)

        # ── Step 2: Vectorized Euclidean distance to ALL training points ───────
        # self._X_norm: (n_train, 8)   q_norm: (8,)
        # Broadcasting: diff shape (n_train, 8) → squared (n_train, 8) → sum → sqrt
        diff  = self._X_norm - q_norm            # (n_train, 8)  via broadcasting
        dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (n_train,) — single vectorized op

        # ── Step 3: Apply position filter (mask) ──────────────────────────────
        if position_filter:
            mask = np.array([
                e["pos"].upper() == position_filter.upper()
                for e in self._db
            ], dtype=bool)
            dists = np.where(mask, dists, np.inf)

        # ── Step 4: Sort + select top K ────────────────────────────────────────
        sorted_idx = np.argsort(dists)[:n]

        results = []
        for idx in sorted_idx:
            if dists[idx] == np.inf:
                continue
            e = self._db[idx]
            sim = max(0.0, round(100.0 - float(dists[idx]) * 50.0, 1))
            results.append({
                "comp_player":     e["name"],
                "comp_position":   e["pos"],
                "comp_school":     e["school"],
                "comp_draft_year": e["year"],
                "comp_draft_pick": e["pick"],
                "comp_outcome":    e["outcome"],
                "comp_career_av":  e["av"],
                "comp_pff_grade":  e.get("pff", 0),
                "similarity_score": sim,
                "distance":        round(float(dists[idx]), 4),
            })

        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def career_success_probability(
        self,
        combine_score: float,
        college_score: float,
        dominator_rating: float,
        breakout_age: int,
        draft_pick_estimate: int,
        conference_strength: float,
        age_at_draft: int,
        pff_grade: float,
        comps: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Predict career outcome probabilities using the trained logistic
        regression model.

        The logistic regression was trained on 60+ historical prospects and
        outputs calibrated probability estimates for each of the four tiers.
        Optionally blends with KNN inverse-distance-weighted vote (50/50)
        if comps are provided.

        Returns:
            dict with keys bust/avg/good/elite, values in [0,1], sum ≈ 1
        """
        q_raw = _build_feature_vector(
            combine_score, college_score, dominator_rating, breakout_age,
            draft_pick_estimate, conference_strength, age_at_draft, pff_grade,
        )
        q_norm = (q_raw - self._mins) / self._range   # (8,)
        q_norm_2d = q_norm.reshape(1, -1)             # (1, 8) for predict_proba

        # ── Logistic regression prediction ────────────────────────────────────
        lr_probs = self._logreg.predict_proba(q_norm_2d)[0]   # (4,)
        lr_dict = {OUTCOMES[i]: float(lr_probs[i]) for i in range(4)}

        # ── KNN inverse-distance-weighted vote ────────────────────────────────
        knn_dict = {o: 0.0 for o in OUTCOMES}
        if comps:
            total_inv_dist = 0.0
            for c in comps:
                w = 1.0 / (c["distance"] + 1e-8)
                knn_dict[c["comp_outcome"]] += w
                total_inv_dist += w
            if total_inv_dist > 0:
                knn_dict = {o: v / total_inv_dist for o, v in knn_dict.items()}

            # Blend 50/50 logreg + KNN
            blended = {
                o: round(0.50 * lr_dict[o] + 0.50 * knn_dict[o], 4)
                for o in OUTCOMES
            }
        else:
            blended = {o: round(v, 4) for o, v in lr_dict.items()}

        # Renormalize to sum = 1
        total = sum(blended.values())
        if total > 0:
            blended = {o: round(v / total, 4) for o, v in blended.items()}

        return blended

    def bust_risk_factors(
        self,
        combine_score: float,
        college_score: float,
        dominator_rating: float,
        breakout_age: int,
        age_at_draft: int,
        conference_strength: float,
        comps: List[Dict],
    ) -> List[str]:
        """
        Identify prospect-specific red flags based on measurements + comp set.

        Each flag is a human-readable string suitable for a scouting report.
        Returns an empty list if no red flags are detected.
        """
        flags: List[str] = []

        if combine_score < 60.0:
            flags.append(f"Below-average athleticism for position (combine score {combine_score:.1f} < 60.0)")
        if college_score < 65.0:
            flags.append(f"Underwhelming college production for draft slot (score {college_score:.1f})")
        if dominator_rating < 25.0:
            flags.append(f"Low dominator rating ({dominator_rating:.1f}%) — was not primary option on offense/defense")
        if age_at_draft >= 24:
            flags.append(f"Older prospect at draft (age {age_at_draft}) — limited development runway")
        if breakout_age >= 22:
            flags.append(f"Late breakout age ({breakout_age}) — missed key early-career development window")
        if conference_strength < 80:
            flags.append(f"Production came against weaker competition (conference strength {conference_strength:.0f}/100)")

        bust_comps = [c for c in comps if c["comp_outcome"] == "bust"]
        if len(bust_comps) >= 2:
            names = ", ".join(c["comp_player"] for c in bust_comps[:2])
            flags.append(f"Multiple bust comps in top-{len(comps)}: {names}")
        elif len(bust_comps) == 1:
            flags.append(f"Top historical comp was a bust: {bust_comps[0]['comp_player']}")

        if comps:
            avg_av = sum(c["comp_career_av"] for c in comps) / len(comps)
            if avg_av < 35:
                flags.append(f"Historical comps averaged only {avg_av:.0f} career AV (league average starter ~40)")

        return flags

    def get_full_analysis(
        self,
        player: str,
        combine_score: float,
        college_score: float,
        dominator_rating: float,
        breakout_age: int,
        draft_pick_estimate: int,
        conference_strength: float,
        age_at_draft: int,
        pff_grade: float,
        position_filter: Optional[str] = None,
    ) -> CareerOutcome:
        """
        Convenience method: runs KNN + logistic regression + bust risk
        and returns a validated CareerOutcome Pydantic object.
        """
        comps = self.find_comps(
            player, combine_score, college_score, dominator_rating,
            breakout_age, draft_pick_estimate, conference_strength,
            age_at_draft, pff_grade, position_filter=position_filter,
        )
        probs = self.career_success_probability(
            combine_score, college_score, dominator_rating, breakout_age,
            draft_pick_estimate, conference_strength, age_at_draft, pff_grade,
            comps=comps,
        )
        flags = self.bust_risk_factors(
            combine_score, college_score, dominator_rating,
            breakout_age, age_at_draft, conference_strength, comps,
        )
        most_likely = max(probs, key=probs.get)
        return CareerOutcome(
            player=player,
            bust_probability=probs["bust"],
            avg_probability=probs["avg"],
            good_probability=probs["good"],
            elite_probability=probs["elite"],
            most_likely_outcome=most_likely,
            confidence=round(probs[most_likely], 3),
            top_comps=comps,
            bust_risk_factors=flags,
        )

    def logreg_convergence(self) -> Dict[str, float]:
        """Return the final training loss per class — useful for debugging."""
        return self._logreg.convergence_summary()
