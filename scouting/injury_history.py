"""
scouting/injury_history.py
════════════════════════════════════════════════════════════════════════════════
Prospect Injury Risk and Durability Scoring

BUSINESS SUMMARY
────────────────
Injury history is not a binary "hurt / not hurt" flag — it is a nuanced risk
profile. A linebacker who tore his ACL as a freshman and returned stronger
is a different risk than one who missed three games per year with soft-tissue
issues. This module scores durability on a 0-100 scale, then applies a
position-specific injury penalty to the base grade.

KEY INSIGHT: position-specific injury weighting
    OL knee/ankle injuries → far more disqualifying than skill positions
    (offensive linemen cannot compensate for knee instability; WRs can adjust)
    ACL for DB/RB → serious but recoverable
    Shoulder for QB → career-altering; flagged heavily
    General soft-tissue history → pattern risk, not single-event risk

SCORING MODEL
─────────────
    durability_score = BASE_DURABILITY - Σ(injury_deductions × position_multiplier)
    injury_adjusted_grade = base_grade × (0.70 + 0.30 × durability_score/100)

    This means a player with durability_score=0 retains 70% of their base grade
    (they are still talented — just injury-prone). A durability_score=100 player
    has the full grade multiplied by 1.0 (no adjustment). This intentional floor
    prevents injury history from zeroing out elite athletic talent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Injury Severity Deductions ────────────────────────────────────────────────

INJURY_DEDUCTIONS = {
    "torn_acl":              25.0,   # ACL — most common career-altering
    "torn_acl_bilateral":    45.0,   # Both ACLs — rare, career risk is extreme
    "shoulder_labrum":       20.0,   # Labrum — surgery required, motion limited
    "shoulder_rotator_cuff": 25.0,   # Rotator cuff — particularly dangerous for QBs
    "broken_foot":           12.0,   # Lisfranc/Jones fracture — chronic recurrence risk
    "stress_fracture":       10.0,   # Bone health concern
    "hamstring_chronic":     15.0,   # Chronic hamstring = will recur
    "concussion_history":    18.0,   # Multiple concussions — NFL durability concern
    "single_concussion":      5.0,   # One documented concussion — modest flag
    "sports_hernia":          8.0,   # Often lingering, performance impact
    "ankle_chronic":         10.0,   # Chronic ankle = explosion risk
    "ankle_high":            12.0,   # High ankle sprain — longer recovery than typical
    "wrist_injury":           7.0,   # Relevant for QB/WR
    "back_injury":           15.0,   # Disc/back — very serious for long-term career
    "knee_meniscus":         12.0,   # Meniscus surgery — cartilage wear concern
    "hip_injury":            10.0,   # Hip health key for RB/OL longevity
    "games_missed_per_year":  3.0,   # Per game missed on average — pattern risk
}

# ── Position Multipliers ──────────────────────────────────────────────────────
# How much each injury type is amplified based on position demands.
# Values > 1.0 = more disqualifying for that position.

POSITION_INJURY_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "QB": {
        "shoulder_rotator_cuff": 2.0,  # Career-defining for QBs
        "shoulder_labrum":       1.8,
        "wrist_injury":          1.5,
        "concussion_history":    1.5,
        "torn_acl":              0.8,  # Less impactful for pocket QBs
        "default":               1.0,
    },
    "OT": {
        "torn_acl":              1.8,  # Pass set requires lateral push
        "knee_meniscus":         1.6,  # Knee health = OL career
        "ankle_chronic":         1.5,
        "hip_injury":            1.6,
        "shoulder_labrum":       1.4,
        "default":               1.0,
    },
    "OG": {
        "torn_acl":              1.7,
        "knee_meniscus":         1.6,
        "hip_injury":            1.5,
        "default":               1.0,
    },
    "RB": {
        "torn_acl":              1.4,  # High recurrence risk at this position
        "hamstring_chronic":     1.6,
        "hip_injury":            1.4,
        "ankle_chronic":         1.3,
        "default":               1.0,
    },
    "WR": {
        "hamstring_chronic":     1.5,
        "ankle_high":            1.3,
        "concussion_history":    1.2,
        "default":               0.9,  # WRs slightly more recoverable
    },
    "TE": {
        "torn_acl":              1.3,
        "shoulder_labrum":       1.3,
        "default":               1.0,
    },
    "EDGE": {
        "torn_acl":              1.3,
        "shoulder_labrum":       1.4,
        "back_injury":           1.5,
        "default":               1.0,
    },
    "DT": {
        "torn_acl":              1.6,
        "hip_injury":            1.5,
        "knee_meniscus":         1.5,
        "back_injury":           1.4,
        "default":               1.0,
    },
    "LB": {
        "concussion_history":    1.4,
        "torn_acl":              1.2,
        "hamstring_chronic":     1.3,
        "default":               1.0,
    },
    "CB": {
        "hamstring_chronic":     1.5,
        "ankle_high":            1.3,
        "concussion_history":    1.2,
        "default":               0.9,
    },
    "S": {
        "concussion_history":    1.6,  # Safety — contact position, long-term concern
        "hamstring_chronic":     1.3,
        "default":               1.0,
    },
}

# ── Hardcoded Injury Database ─────────────────────────────────────────────────

@dataclass
class InjuryRecord:
    player: str
    position: str
    # Injury flags
    torn_acl:              bool  = False
    torn_acl_bilateral:    bool  = False
    shoulder_labrum:       bool  = False
    shoulder_rotator_cuff: bool  = False
    broken_foot:           bool  = False
    stress_fracture:       bool  = False
    hamstring_chronic:     bool  = False
    concussion_history:    bool  = False
    single_concussion:     bool  = False
    sports_hernia:         bool  = False
    ankle_chronic:         bool  = False
    ankle_high:            bool  = False
    wrist_injury:          bool  = False
    back_injury:           bool  = False
    knee_meniscus:         bool  = False
    hip_injury:            bool  = False
    # Average games missed per college season (0 = perfect health)
    games_missed_per_year: float = 0.0
    # Free text notes
    notes: str = ""


INJURY_DATABASE: Dict[str, InjuryRecord] = {
    # ── 2025 Draft Class ──────────────────────────────────────────────────────
    "Cam Ward":          InjuryRecord("Cam Ward", "QB", notes="No significant injury history. Clean bill of health entering draft."),
    "Travis Hunter":     InjuryRecord("Travis Hunter", "CB", notes="Minor shoulder stinger in 2023, not surgery. Fully cleared."),
    "Abdul Carter":      InjuryRecord("Abdul Carter", "EDGE", notes="No significant injuries. Model prospect for durability."),
    "Will Campbell":     InjuryRecord("Will Campbell", "OT", notes="Ankle sprain 2023, missed 1 game. No structural concern."),
    "Mason Graham":      InjuryRecord("Mason Graham", "DT", notes="Clean health record across entire Michigan career."),
    "Tetairoa McMillan": InjuryRecord("Tetairoa McMillan", "WR", notes="No injuries of note. Three full seasons of production."),
    "Malaki Starks":     InjuryRecord("Malaki Starks", "S", notes="Played through minor injuries; no surgery on record."),
    "Jalon Walker":      InjuryRecord("Jalon Walker", "EDGE", notes="Healthy. Played all 13 games in 2024."),
    "Darius Alexander":  InjuryRecord("Darius Alexander", "DT", notes="Clean durability record at Toledo."),
    "Mykel Williams":    InjuryRecord("Mykel Williams", "EDGE", games_missed_per_year=1.5, notes="Missed 2 games in 2023 with hamstring; returned at full speed."),
    "Jihaad Campbell":   InjuryRecord("Jihaad Campbell", "LB", torn_acl=True, games_missed_per_year=2.0,
                                       notes="ACL tear November 2022, missed full 2022 season. Returned 100% in 2023. Alabama training staff noted exceptional recovery."),
    "Kelvin Banks Jr":   InjuryRecord("Kelvin Banks Jr", "OT", notes="No injuries. Five full semesters of starting experience."),
    "Tyler Warren":      InjuryRecord("Tyler Warren", "TE", notes="Walk-on durability — played through multiple minor issues without missing games."),
    "Shemar Stewart":    InjuryRecord("Shemar Stewart", "EDGE", notes="No significant injuries at Texas A&M."),
    "Omarion Hampton":   InjuryRecord("Omarion Hampton", "RB", notes="No injuries of concern. Took heavy workload at UNC cleanly."),
    "Luther Burden III": InjuryRecord("Luther Burden III", "WR", notes="Healthy. Clean record."),
    "Shedeur Sanders":   InjuryRecord("Shedeur Sanders", "QB", games_missed_per_year=0.5,
                                       notes="Took significant sack numbers at Colorado; no structural damage reported. Minor shoulder soreness treated conservatively."),
    "Nick Emmanwori":    InjuryRecord("Nick Emmanwori", "S", notes="No injury history at South Carolina."),
    # ── 2024 Draft Class ──────────────────────────────────────────────────────
    "Caleb Williams":    InjuryRecord("Caleb Williams", "QB", games_missed_per_year=1.0,
                                       notes="Missed USC games 2023 with knee soreness. Not structural. Cleared for full combine activities."),
    "Marvin Harrison Jr":InjuryRecord("Marvin Harrison Jr", "WR", notes="No injuries. Full three-year run at Ohio State."),
    "Joe Alt":           InjuryRecord("Joe Alt", "OT", notes="Clean health record at Notre Dame."),
    "Rome Odunze":       InjuryRecord("Rome Odunze", "WR", notes="No injuries. Played 14 games in 2023."),
    "Terrion Arnold":    InjuryRecord("Terrion Arnold", "CB", notes="No injuries at Alabama."),
    # ── 2023 Draft Class ──────────────────────────────────────────────────────
    "Bryce Young":       InjuryRecord("Bryce Young", "QB", games_missed_per_year=0.5,
                                       notes="No surgery on record. Size (5'10\", 204lb) is injury risk in itself at NFL level."),
    "CJ Stroud":         InjuryRecord("CJ Stroud", "QB", notes="Clean health record at Ohio State."),
    "Anthony Richardson":InjuryRecord("Anthony Richardson", "QB", games_missed_per_year=1.5,
                                       notes="Shoulder injury in 2022 limited production. Medical staff cleared for 2023 draft."),
    "Will Anderson Jr":  InjuryRecord("Will Anderson Jr", "EDGE", notes="Clean health record at Alabama."),
    "Jalen Carter":      InjuryRecord("Jalen Carter", "DT", games_missed_per_year=3.5,
                                       notes="Missed significant time at Georgia with leg/knee issues. Medical flags raised pre-draft by multiple teams. Concern about long-term availability."),
    # ── 2022 Draft Class ──────────────────────────────────────────────────────
    "Aidan Hutchinson":  InjuryRecord("Aidan Hutchinson", "EDGE", notes="Healthy. Played 14 games in 2021 senior season."),
    "Travon Walker":     InjuryRecord("Travon Walker", "EDGE", notes="No significant injury history at Georgia."),
    "Ahmad Gardner":     InjuryRecord("Ahmad Gardner", "CB", notes="Three seasons at Cincinnati with no missed games."),
    "Kyle Hamilton":     InjuryRecord("Kyle Hamilton", "S", games_missed_per_year=1.0,
                                       notes="Knee scope 2022 — relatively minor. Cleared by Ravens medical staff."),
    # ── Historical Reference ──────────────────────────────────────────────────
    "Patrick Mahomes":   InjuryRecord("Patrick Mahomes", "QB", notes="No college injuries of significance."),
    "Justin Jefferson":  InjuryRecord("Justin Jefferson", "WR", notes="Healthy at LSU across two seasons as primary target."),
    "JaMarcus Russell":  InjuryRecord("JaMarcus Russell", "QB", notes="No college injury history. Physical decline was conditioning-related."),
    "Johnny Manziel":    InjuryRecord("Johnny Manziel", "QB", notes="No significant injuries at Texas A&M."),
    "Josh Rosen":        InjuryRecord("Josh Rosen", "QB", shoulder_labrum=True,
                                       notes="Labrum surgery before 2018 draft. Fully disclosed. Shoulder health flag for OBP."),
}


class ProspectInjuryAnalyzer:
    """
    Scores prospect injury history and adjusts draft grades accordingly.

    Position-specific multipliers amplify injuries that are disproportionately
    disqualifying at certain positions (e.g., knee injuries for OL, shoulder
    for QB).

    The adjustment function intentionally preserves a floor: even a catastrophic
    injury history reduces the grade to at most 70% of base — because raw talent
    still matters. This mirrors how NFL teams actually value injured prospects:
    they discount but do not discard.
    """

    def __init__(self) -> None:
        self._db = INJURY_DATABASE

    # ── Public API ────────────────────────────────────────────────────────────

    def durability_score(self, player: str) -> float:
        """
        Return a 0-100 durability score. 100 = perfect health, 0 = catastrophic.

        Each injury flag applies a deduction, amplified by the position-specific
        multiplier for that injury type. Average games missed per year adds a
        separate pattern-risk deduction.

        Returns: float in [0, 100]
        """
        record = self._resolve(player)
        pos = record.position.upper()
        multipliers = POSITION_INJURY_MULTIPLIERS.get(pos, {})
        default_mult = multipliers.get("default", 1.0)

        total_deduction = 0.0

        # Apply each boolean injury flag
        for injury_key, base_ded in INJURY_DEDUCTIONS.items():
            if injury_key == "games_missed_per_year":
                continue  # Handled separately
            flag = getattr(record, injury_key, False)
            if flag:
                mult = multipliers.get(injury_key, default_mult)
                total_deduction += base_ded * mult

        # Games missed deduction (per game × position multiplier)
        games_ded = record.games_missed_per_year * INJURY_DEDUCTIONS["games_missed_per_year"]
        total_deduction += games_ded * default_mult

        return round(max(0.0, min(100.0, 100.0 - total_deduction)), 1)

    def injury_adjusted_grade(self, player: str, base_grade: float) -> float:
        """
        Adjust a base draft grade downward based on injury history.

        Formula:
            adjustment_factor = 0.70 + 0.30 × (durability_score / 100)
            adjusted_grade    = base_grade × adjustment_factor

        This means:
            durability=100 → factor=1.00 (no adjustment)
            durability= 50 → factor=0.85 (15% haircut)
            durability=  0 → factor=0.70 (30% maximum haircut)

        The 0.70 floor ensures a physically elite prospect with serious injury
        history still projects meaningfully — matching how NFL teams behave.

        Args:
            player:     Player name
            base_grade: Draft grade before injury adjustment (0-100)

        Returns:
            float in [0, 100]
        """
        dur = self.durability_score(player)
        factor = 0.70 + 0.30 * (dur / 100.0)
        return round(min(100.0, base_grade * factor), 1)

    def risk_summary(self, player: str) -> Dict:
        """
        Return a full injury risk summary dict for a player.
        """
        record = self._resolve(player)
        dur = self.durability_score(player)
        flags = self._get_flags(record)
        risk_level = (
            "CLEAN"    if dur >= 90 else
            "LOW"      if dur >= 75 else
            "MODERATE" if dur >= 55 else
            "HIGH"     if dur >= 35 else
            "CRITICAL"
        )
        return {
            "player":              player,
            "position":            record.position,
            "durability_score":    dur,
            "risk_level":          risk_level,
            "active_flags":        flags,
            "games_missed_per_year": record.games_missed_per_year,
            "notes":               record.notes,
        }

    def all_players(self) -> List[str]:
        return list(self._db.keys())

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _resolve(self, player: str) -> InjuryRecord:
        if player in self._db:
            return self._db[player]
        return InjuryRecord(player=player, position="UNK",
                             notes="No injury data on file.")

    def _get_flags(self, record: InjuryRecord) -> List[str]:
        flags = []
        for key in INJURY_DEDUCTIONS:
            if key == "games_missed_per_year":
                if record.games_missed_per_year > 1.0:
                    flags.append(f"games_missed_per_year: {record.games_missed_per_year:.1f}")
            elif getattr(record, key, False):
                flags.append(key)
        return flags
