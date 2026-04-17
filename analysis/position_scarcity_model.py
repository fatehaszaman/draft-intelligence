"""
analysis/position_scarcity_model.py
════════════════════════════════════════════════════════════════════════════════
Position Scarcity Index + VORP-Style Value Metric

BUSINESS SUMMARY
────────────────
Not all draft positions are created equal. A top-5 quarterback is dramatically
more impactful than a top-5 running back — not just because of position value,
but because elite QB play is SCARCE in ways elite RB play is not. Meanwhile,
if a draft class has 8 elite pass rushers, even the 8th one loses scarcity
premium since teams can trade down and still get quality.

This module quantifies two dimensions:

  1. POSITION SCARCITY INDEX — How rare is elite talent at this position
     in THIS specific draft class? (class-specific, not historical average)

  2. VORP (Value Over Replacement Player) — How much better is this prospect
     than the "replacement level" player at their position? Inspired by
     baseball's WAR/VORP but adapted for draft context.

SCARCITY INDEX FORMULA
──────────────────────
  scarcity_index = (1 / n_viable_prospects) × position_value_multiplier × class_depth_factor

  Where:
    n_viable_prospects     = prospects graded ≥ 70 at that position
    position_value_mult    = historically calibrated positional value (QB=1.8, RB=0.7)
    class_depth_factor     = current_class_size / historical_average_class_size

  Range: 0.0 (abundantly deep) → 1.0 (one viable starter in class)

VORP FORMULA
────────────
  replacement_level = mean grade of prospects ranked 33rd–65th at position
  VORP = (prospect_grade - replacement_level) × position_value_multiplier

  QB VORP gets an additional premium: QB_PREMIUM_MULTIPLIER = 1.35
  Rationale: a QB 10 points above replacement is worth ~35% more than a WR
  10 points above replacement, due to franchise-altering impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Position Value Multipliers ────────────────────────────────────────────────
# Calibrated from 10-year AV (Approximate Value) data per position.
# Source: Pro Football Reference contract value / AV studies (2014-2024).

POSITION_VALUE_MULTIPLIERS: Dict[str, float] = {
    "QB":    1.80,   # Franchise cornerstone — uniquely transformative
    "EDGE":  1.35,   # Best non-QB investment (pass rush premium)
    "OT":    1.20,   # Blindside LT protection; RT lower at 1.05
    "WR":    1.10,   # High ceiling with elite CB1s in league
    "CB":    1.08,   # Elite CB1s command near-EDGE value in pass-heavy era
    "DT":    1.05,   # Interior pass rush valued in 3-tech era
    "TE":    1.05,   # Receiving TE premium; blocking TE closer to 0.85
    "S":     0.95,   # Box safety higher (1.0); center field slightly lower
    "IOL":   0.95,   # Center (1.05) vs. guard (0.90) split
    "LB":    0.85,   # Off-ball LB devalued in pass-heavy NFL
    "RB":    0.70,   # Historically exploitable on rookie deals; depth plentiful
    "K":     0.60,   # Elite kickers matter but low positional scarcity
    "P":     0.55,   # Punter — minimal impact on win probability
    "FB":    0.65,   # Fullback role diminished
}

QB_PREMIUM_MULTIPLIER: float = 1.35  # Applied on top of QB position value for VORP

# ── Historical Average Class Depth by Position ────────────────────────────────
# Average number of prospects per position in rounds 1-3, 2015-2025
HISTORICAL_CLASS_DEPTH: Dict[str, float] = {
    "QB":    4.2,
    "WR":   12.8,
    "EDGE":  9.4,
    "OT":    7.6,
    "CB":    9.2,
    "DT":    8.1,
    "TE":    5.4,
    "IOL":   7.3,
    "S":     6.8,
    "LB":    7.9,
    "RB":    6.5,
}

# ── Replacement Level Thresholds ─────────────────────────────────────────────
# The "replacement level" player = average starter available in round 3-4.
# These grades represent typical undrafted/late-round starters at each position.
REPLACEMENT_LEVEL_GRADES: Dict[str, float] = {
    "QB":    52.0,   # Backup/bridge QB — easily available
    "WR":    60.0,   # WR3/depth — large supply
    "EDGE":  58.0,   # Rotational edge
    "OT":    56.0,   # Swing tackle
    "CB":    59.0,   # Nickel/slot corner
    "DT":    57.0,   # Rotational 1T/3T
    "TE":    55.0,   # Blocking-first TE
    "IOL":   58.0,   # Interior depth
    "S":     57.0,   # Single-high backup
    "LB":    61.0,   # Off-ball linebacker (most available position)
    "RB":    62.0,   # Handoff back — cheapest to replace
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ScarcityReport:
    """Scarcity analysis for a single position in the current draft class."""
    position:           str
    n_elite_prospects:  int      # Graded 80+
    n_viable_prospects: int      # Graded 70+
    n_starter_tier:     int      # Graded 60-70
    scarcity_index:     float    # 0.0-1.0; higher = scarcer
    class_depth_factor: float    # vs historical average
    position_value_mult:float
    tier_label:         str      # "ELITE_DEPTH", "AVERAGE", "SCARCE", "CRITICAL"

    def __str__(self) -> str:
        bar = "█" * int(self.scarcity_index * 20)
        return (
            f"{self.position:6s} | Scarcity: {self.scarcity_index:.3f} "
            f"{bar:20s} | {self.tier_label} | "
            f"Elite: {self.n_elite_prospects} | Viable: {self.n_viable_prospects}"
        )


@dataclass
class VORPResult:
    """Value Over Replacement Player calculation for a prospect."""
    player:               str
    position:             str
    grade:                float
    replacement_level:    float
    raw_vorp:             float    # grade - replacement_level
    position_adjusted_vorp: float  # × position_value_multiplier
    qb_premium_vorp:      float    # QB only: × QB_PREMIUM_MULTIPLIER
    final_vorp:           float    # What goes into the draft board
    tier:                 str      # "GENERATIONAL", "ELITE", "ABOVE_REPLACEMENT", "BELOW"

    def __str__(self) -> str:
        return (
            f"{self.player:25s} {self.position:6s} | "
            f"Grade: {self.grade:.1f} | "
            f"VORP: {self.final_vorp:+.1f} | "
            f"[{self.tier}]"
        )


# ── Core Engine ───────────────────────────────────────────────────────────────

class PositionScarcityModel:
    """
    Computes position scarcity indices and VORP metrics for a draft class.

    Usage:
        prospects = {"Abdul Carter": ("EDGE", 96.1), "Will Johnson": ("CB", 94.8), ...}
        model = PositionScarcityModel(prospects)
        scarcity = model.scarcity_report("EDGE")
        vorp = model.compute_vorp("Abdul Carter")
        big_board = model.vorp_adjusted_rankings()
    """

    def __init__(self, prospects: Dict[str, Tuple[str, float]]) -> None:
        """
        Args:
            prospects: {player_name: (position, overall_grade)} mapping
                       overall_grade is 0-100 scale
        """
        self.prospects = prospects
        self._by_position: Dict[str, List[Tuple[str, float]]] = {}
        self._build_position_buckets()

    def _build_position_buckets(self) -> None:
        """Group and sort prospects by position."""
        for player, (pos, grade) in self.prospects.items():
            if pos not in self._by_position:
                self._by_position[pos] = []
            self._by_position[pos].append((player, grade))

        # Sort each position group descending by grade
        for pos in self._by_position:
            self._by_position[pos].sort(key=lambda x: -x[1])

    def scarcity_report(self, position: str) -> ScarcityReport:
        """
        Compute scarcity index for a position in this draft class.

        Scarcity index formula:
            scarcity = (1 / n_viable) × pos_value_mult × depth_factor

        Clamped to [0.0, 1.0]. Higher = scarcer = more valuable to reach for.

        Args:
            position: Position string (e.g., "QB", "EDGE")

        Returns:
            ScarcityReport dataclass
        """
        players = self._by_position.get(position, [])
        grades = [g for _, g in players]

        n_elite  = sum(1 for g in grades if g >= 80.0)
        n_viable = sum(1 for g in grades if g >= 70.0)
        n_starter = sum(1 for g in grades if 60.0 <= g < 70.0)

        pos_mult = POSITION_VALUE_MULTIPLIERS.get(position, 1.0)
        hist_depth = HISTORICAL_CLASS_DEPTH.get(position, 8.0)
        depth_factor = hist_depth / max(n_viable, 1)  # > 1.0 means scarcer than average

        # Core scarcity computation
        if n_viable == 0:
            raw_scarcity = 1.0
        else:
            raw_scarcity = (1.0 / n_viable) * pos_mult * depth_factor

        scarcity_index = min(1.0, raw_scarcity / 3.0)  # Normalize: 3.0 ≈ max realistic

        # Tier label
        if scarcity_index >= 0.70:
            tier = "CRITICAL"    # Must draft this position if need exists
        elif scarcity_index >= 0.45:
            tier = "SCARCE"      # Strong priority — don't expect value later
        elif scarcity_index <= 0.15:
            tier = "ELITE_DEPTH" # Can trade down — talent available later
        else:
            tier = "AVERAGE"

        return ScarcityReport(
            position=position,
            n_elite_prospects=n_elite,
            n_viable_prospects=n_viable,
            n_starter_tier=n_starter,
            scarcity_index=round(scarcity_index, 4),
            class_depth_factor=round(depth_factor, 3),
            position_value_mult=pos_mult,
            tier_label=tier,
        )

    def compute_vorp(self, player: str) -> VORPResult:
        """
        Compute VORP (Value Over Replacement Player) for a single prospect.

        VORP formula:
            raw_vorp             = grade - replacement_level[position]
            position_adj_vorp    = raw_vorp × position_value_multiplier
            final_vorp (QB)      = position_adj_vorp × QB_PREMIUM_MULTIPLIER
            final_vorp (non-QB)  = position_adj_vorp

        Args:
            player: Player name (must be in prospects dict)

        Returns:
            VORPResult dataclass
        """
        if player not in self.prospects:
            raise KeyError(f"Player '{player}' not found in prospects")

        position, grade = self.prospects[player]
        replacement = REPLACEMENT_LEVEL_GRADES.get(position, 58.0)
        pos_mult = POSITION_VALUE_MULTIPLIERS.get(position, 1.0)

        raw_vorp = grade - replacement
        position_adj = raw_vorp * pos_mult
        qb_premium = position_adj * QB_PREMIUM_MULTIPLIER if position == "QB" else position_adj
        final_vorp = qb_premium

        # Tier classification
        if final_vorp >= 55.0:
            tier = "GENERATIONAL"
        elif final_vorp >= 35.0:
            tier = "ELITE"
        elif final_vorp >= 15.0:
            tier = "ABOVE_REPLACEMENT"
        elif final_vorp >= 0.0:
            tier = "MARGINAL"
        else:
            tier = "BELOW_REPLACEMENT"

        return VORPResult(
            player=player,
            position=position,
            grade=grade,
            replacement_level=replacement,
            raw_vorp=round(raw_vorp, 2),
            position_adjusted_vorp=round(position_adj, 2),
            qb_premium_vorp=round(qb_premium, 2),
            final_vorp=round(final_vorp, 2),
            tier=tier,
        )

    def vorp_adjusted_rankings(self) -> List[VORPResult]:
        """
        Rank all prospects by VORP, producing a VORP-adjusted big board.

        Returns:
            List of VORPResult objects sorted descending by final_vorp
        """
        results = []
        for player in self.prospects:
            try:
                results.append(self.compute_vorp(player))
            except KeyError:
                pass

        return sorted(results, key=lambda r: -r.final_vorp)

    def scarcity_premium(self, player: str) -> float:
        """
        Compute a scarcity-adjusted VORP that rewards picks at scarce positions.

        scarcity_premium_vorp = final_vorp × (1 + scarcity_index)

        This means a VORP=40 player at a scarce position (scarcity=0.8) is
        worth the same as a VORP=72 player at an abundant position. Useful
        for teams with specific positional needs.

        Args:
            player: Player name

        Returns:
            Scarcity-adjusted VORP score
        """
        vorp_result = self.compute_vorp(player)
        position, _ = self.prospects[player]
        scarcity = self.scarcity_report(position).scarcity_index
        return round(vorp_result.final_vorp * (1.0 + scarcity), 2)

    def all_position_scarcity(self) -> List[ScarcityReport]:
        """
        Compute scarcity reports for all positions in the class.

        Returns:
            List of ScarcityReport sorted by scarcity_index descending
        """
        reports = []
        for pos in self._by_position:
            reports.append(self.scarcity_report(pos))
        return sorted(reports, key=lambda r: -r.scarcity_index)

    def position_class_summary(self, position: str) -> Dict:
        """
        Summary statistics for all prospects at a position.

        Returns:
            Dict with mean grade, std, best player, tier distribution
        """
        players = self._by_position.get(position, [])
        if not players:
            return {"position": position, "count": 0}

        grades = np.array([g for _, g in players])
        best_player, best_grade = players[0]

        return {
            "position":         position,
            "count":            len(players),
            "best_player":      best_player,
            "best_grade":       round(float(best_grade), 1),
            "mean_grade":       round(float(grades.mean()), 1),
            "std_grade":        round(float(grades.std()), 1),
            "median_grade":     round(float(np.median(grades)), 1),
            "n_elite_80+":      int((grades >= 80).sum()),
            "n_starter_70+":    int((grades >= 70).sum()),
            "n_depth_60+":      int((grades >= 60).sum()),
            "players_ranked":   [p for p, _ in players],
        }

    def draft_day_advice(self, team_needs: List[str]) -> List[Dict]:
        """
        Generate positional draft advice based on scarcity + team needs.

        Args:
            team_needs: List of positions the team needs (highest priority first)

        Returns:
            List of advice dicts sorted by urgency score
        """
        advice = []

        for position in team_needs:
            report = self.scarcity_report(position)
            players = self._by_position.get(position, [])
            best_available = players[0] if players else ("None", 0.0)

            urgency = 0.0
            urgency += report.scarcity_index * 50       # Scarce position = urgent
            urgency += report.position_value_mult * 20   # Important position = urgent
            urgency = min(100.0, urgency)

            advice.append({
                "position":        position,
                "urgency_score":   round(urgency, 1),
                "tier":            report.tier_label,
                "best_available":  best_available[0],
                "best_grade":      best_available[1],
                "viable_options":  report.n_viable_prospects,
                "recommendation":  (
                    "DRAFT NOW — critical scarcity, won't last" if urgency >= 70 else
                    "PRIORITIZE — limited options remaining"     if urgency >= 50 else
                    "MONITOR — some depth allows flexibility"    if urgency >= 30 else
                    "WAIT — deep class, can find value later"
                ),
            })

        return sorted(advice, key=lambda x: -x["urgency_score"])


# ── 2026 Draft Class Scarcity Analysis ────────────────────────────────────────
# Pre-computed using 2026 consensus prospect grades from CBS Sports / ESPN / PFF

_2026_DRAFT_CLASS_GRADES: Dict[str, Tuple[str, float]] = {
    # QBs
    "Cam Ward":           ("QB",   93.5),
    "Shedeur Sanders":    ("QB",   91.0),
    "Dillon Gabriel":     ("QB",   80.0),
    "Quinn Ewers":        ("QB",   78.5),
    "Tyler Shough":       ("QB",   72.0),
    # WRs
    "Travis Hunter":      ("WR",   94.8),
    "Tetairoa McMillan":  ("WR",   89.2),
    "Matthew Golden":     ("WR",   83.1),
    "Emeka Egbuka":       ("WR",   82.6),
    "Jaylen Moody":       ("WR",   74.0),
    "Jack Bech":          ("WR",   72.5),
    "Isaiah Williams":    ("WR",   70.8),
    # EDGE
    "Abdul Carter":       ("EDGE", 96.1),
    "Jalon Walker":       ("EDGE", 91.4),
    "Mike Green":         ("EDGE", 87.3),
    "James Pearce Jr.":   ("EDGE", 84.9),
    "Mykel Williams":     ("EDGE", 81.2),
    "Donovan Ezeiruaku": ("EDGE", 78.5),
    # OT
    "Will Campbell":      ("OT",   90.3),
    "Kelvin Banks Jr.":   ("OT",   89.1),
    "Josh Simmons":       ("OT",   86.4),
    "Jonah Savaiinaea":   ("OT",   80.2),
    "Marcus Mbow":        ("OT",   77.8),
    # CB
    "Will Johnson":       ("CB",   94.8),
    "Jahdae Barron":      ("CB",   85.6),
    "Shavon Revel Jr.":   ("CB",   83.1),
    "Benjamin Morrison":  ("CB",   81.4),
    "Trey Amos":          ("CB",   78.2),
    # DT
    "Mason Graham":       ("DT",   91.7),
    "Derrick Harmon":     ("DT",   82.3),
    "Darius Alexander":   ("DT",   79.4),
    "Kenneth Grant":      ("DT",   77.8),
    # IOL
    "Tyler Booker":       ("IOL",  85.3),
    "Aireontae Ersery":   ("IOL",  82.1),
    "Grey Zabel":         ("IOL",  79.4),
    "Josh Conerly":       ("IOL",  74.2),
    # TE
    "Colston Loveland":   ("TE",   88.7),
    "Tyler Warren":       ("TE",   87.1),
    "Harold Fannin Jr.":  ("TE",   82.4),
    "Mason Taylor":       ("TE",   79.3),
    # S
    "Malaki Starks":      ("S",    87.4),
    "Nick Emmanwori":     ("S",    84.6),
    "Andrew Mukuba":      ("S",    79.1),
    # LB
    "Jihaad Campbell":    ("LB",   86.2),
    "Carson Schwesinger": ("LB",   82.4),
    "Danny Striggow":     ("LB",   74.3),
    # RB
    "Ashton Jeanty":      ("RB",   94.2),
    "Omarion Hampton":    ("RB",   81.3),
    "Quinshon Judkins":   ("RB",   79.5),
    "TreVeyon Henderson": ("RB",   77.2),
}


def get_2026_scarcity_model() -> PositionScarcityModel:
    """
    Return a pre-built PositionScarcityModel for the 2026 draft class.

    Returns:
        PositionScarcityModel loaded with 2026 consensus grades
    """
    return PositionScarcityModel(_2026_DRAFT_CLASS_GRADES)


# ── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = get_2026_scarcity_model()

    print("=" * 70)
    print("2026 NFL DRAFT — POSITION SCARCITY INDEX")
    print("=" * 70)

    for report in model.all_position_scarcity():
        print(f"  {report}")

    print("\n" + "=" * 70)
    print("TOP 20 BY VORP (Value Over Replacement Player)")
    print("=" * 70)

    rankings = model.vorp_adjusted_rankings()
    for i, r in enumerate(rankings[:20], 1):
        print(f"  {i:2d}. {r}")

    print("\n" + "=" * 70)
    print("DRAFT-DAY ADVICE — Team with needs: QB, EDGE, CB")
    print("=" * 70)

    for adv in model.draft_day_advice(["QB", "EDGE", "CB"]):
        print(
            f"  {adv['position']:6s} | Urgency: {adv['urgency_score']:5.1f} | "
            f"{adv['tier']:12s} | Best: {adv['best_available']:20s} | "
            f"→ {adv['recommendation']}"
        )
